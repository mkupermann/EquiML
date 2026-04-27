"""Feedback-loop simulation: K-round retrain cycle that tracks fairness drift.

See docs/rfcs/0005-feedback-loop-simulation.md.

The classic failure mode this module models: a model is deployed, makes
decisions, those decisions affect the next training set, the next model
trains on a biased corpus, and the bias compounds. Lending, hiring,
predictive-policing, and admissions systems all have this shape.

Limitations (load-bearing — read before using):

- Feedback rules are deterministic simplifications. Real-world feedback
  loops include applicant-pool composition shifts, macroeconomic shocks,
  strategic responses by applicants, and adversarial gaming. None of
  that is here.
- The simulator assumes the held-out test set stays fixed across
  rounds. That isolates the model-drift signal but understates total
  deployment risk; in production the test distribution drifts too.
- "Selection bias" feedback drops rejected applicants entirely. Real
  systems sometimes observe a noisy signal on the rejected pool (a
  hand-reviewed sample, a downstream measurement). That partial-info
  case is future work.
- The simulator runs deterministic feedback rules. It does not pretend
  to be a behavioural simulation of the underlying market.

Use this for what-if reasoning about deployment risk, not for forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision rules: how the model's scores become "selected" decisions
# ---------------------------------------------------------------------------


class DecisionRule(str, Enum):
    """How model scores translate into deployed decisions.

    THRESHOLD: select all applicants with proba > threshold.
    TOP_K:     select the K highest-scoring applicants.
    """

    THRESHOLD = "threshold"
    TOP_K = "top_k"


def _select_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return integer indices selected under a probability threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return np.where(scores > threshold)[0]


def _select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return integer indices of the K highest-scoring entries."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    # argsort gives ascending; flip and take the top k.
    return np.argsort(-scores, kind="stable")[:k]


def apply_decision_rule(
    scores: np.ndarray,
    rule: DecisionRule,
    *,
    threshold: float = 0.5,
    k: int | None = None,
) -> np.ndarray:
    """Apply a decision rule to a score array, return selected indices."""
    if rule == DecisionRule.THRESHOLD:
        return _select_threshold(scores, threshold)
    if rule == DecisionRule.TOP_K:
        if k is None:
            raise ValueError("DecisionRule.TOP_K requires k=...")
        return _select_top_k(scores, k)
    raise ValueError(f"Unsupported decision rule: {rule!r}")


# ---------------------------------------------------------------------------
# Feedback rules: how decisions flow back into the next training set
# ---------------------------------------------------------------------------


class FeedbackRule(str, Enum):
    """How this round's decisions update the next round's training data.

    SELECTION_BIAS:
        Only selected applicants' true outcomes are observed and added to
        the next training set; rejected applicants are dropped entirely.
        The canonical lending / hiring feedback loop.

    PERFECT_INFO:
        All outcomes are observed regardless of decision. Baseline used
        to verify the simulator does not invent drift on its own.

    LABEL_FLIP (future):
        Reserved for a richer simulation where selected applicants who
        succeed retroactively get positive labels in the next round.
    """

    SELECTION_BIAS = "selection_bias"
    PERFECT_INFO = "perfect_info"


def apply_feedback(
    rule: FeedbackRule,
    X_round: pd.DataFrame,
    y_round: pd.Series,
    sensitive_round: pd.Series,
    selected_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, sensitive) rows that survive this feedback rule.

    `selected_idx` is the integer-positional index into X_round for rows
    the model selected this round. The returned frames carry the indices
    of `X_round` so they can be appended to a growing training set.
    """
    if rule == FeedbackRule.PERFECT_INFO:
        return X_round.copy(), y_round.copy(), sensitive_round.copy()
    if rule == FeedbackRule.SELECTION_BIAS:
        if len(selected_idx) == 0:
            empty_X = X_round.iloc[0:0].copy()
            empty_y = y_round.iloc[0:0].copy()
            empty_s = sensitive_round.iloc[0:0].copy()
            return empty_X, empty_y, empty_s
        return (
            X_round.iloc[selected_idx].copy(),
            y_round.iloc[selected_idx].copy(),
            sensitive_round.iloc[selected_idx].copy(),
        )
    raise ValueError(f"Unsupported feedback rule: {rule!r}")


# ---------------------------------------------------------------------------
# Per-round result
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    """Snapshot of one round in a feedback-loop simulation."""

    round_idx: int
    metrics: dict[str, float]
    per_sensitive: dict[str, dict[str, float]]
    n_train: int
    n_selected: int
    selection_rate_per_group: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class FeedbackLoopSimulator:
    """Run a K-round retrain loop and record fairness at each round.

    Each round:
      1. Train a fresh model (from `model_factory`) on the current pool.
      2. Predict on the test set; record DP, EO, accuracy, per-group rates.
      3. Predict on a fresh applicant pool drawn from the same distribution
         as the initial training data.
      4. Apply the decision rule to that pool's scores.
      5. Apply the feedback rule to update the training pool.
      6. Repeat.

    The applicant pool at each round is sampled with replacement from the
    *initial* training set. That is a conservative simplification: a full
    simulation would draw from a population whose composition itself
    drifts. We document this and stop short of inventing one.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        sensitive_features: list[str],
        decision_rule: DecisionRule = DecisionRule.THRESHOLD,
        feedback_rule: FeedbackRule = FeedbackRule.SELECTION_BIAS,
        n_rounds: int = 10,
        random_state: int = 42,
        *,
        threshold: float = 0.5,
        top_k: int | None = None,
        applicants_per_round: int | None = None,
    ):
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        if not sensitive_features:
            raise ValueError("at least one sensitive feature is required")
        self.model_factory = model_factory
        self.sensitive_features = list(sensitive_features)
        self.decision_rule = DecisionRule(decision_rule)
        self.feedback_rule = FeedbackRule(feedback_rule)
        self.n_rounds = int(n_rounds)
        self.random_state = int(random_state)
        self.threshold = float(threshold)
        self.top_k = top_k
        self.applicants_per_round = applicants_per_round

    # -- internals ----------------------------------------------------------

    def _get_scores(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Best-effort score extraction. Falls back to hard predictions."""
        try:
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba, dtype=float)
        except (AttributeError, ValueError, NotImplementedError):
            preds = np.asarray(model.predict(X), dtype=float)
            return preds

    def _per_group_selection_rate(
        self, sensitive: pd.Series, selected_mask: np.ndarray
    ) -> dict[str, float]:
        rates: dict[str, float] = {}
        sensitive_arr = np.asarray(sensitive)
        for group in pd.unique(sensitive_arr):
            mask = sensitive_arr == group
            n = int(mask.sum())
            if n == 0:
                continue
            rates[str(group)] = float(selected_mask[mask].sum()) / n
        return rates

    def _round_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive: pd.Series,
    ) -> dict[str, float]:
        # Fairlearn raises when only one group is present in y_true; guard.
        unique_groups = pd.unique(np.asarray(sensitive))
        if len(unique_groups) < 2:
            dp = 0.0
            eo = 0.0
        else:
            dp = float(
                demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive
                )
            )
            try:
                eo = float(
                    equalized_odds_difference(
                        y_true, y_pred, sensitive_features=sensitive
                    )
                )
            except (ValueError, ZeroDivisionError):
                eo = float("nan")
        acc = float(accuracy_score(y_true, y_pred))
        return {
            "demographic_parity_difference": dp,
            "equalized_odds_difference": eo,
            "accuracy": acc,
        }

    # -- public API ---------------------------------------------------------

    def run(
        self,
        X_train_init: pd.DataFrame,
        y_train_init: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_train_init: pd.Series,
        sensitive_test: pd.Series,
    ) -> list[RoundResult]:
        """Run the simulation and return one RoundResult per round.

        The first sensitive feature in `self.sensitive_features` is the
        primary one used for fairness metrics. Multi-sensitive
        intersectional simulation is future work.
        """
        if X_train_init.empty:
            raise ValueError("X_train_init must not be empty")
        if len(X_train_init) != len(y_train_init):
            raise ValueError("X_train_init and y_train_init length mismatch")
        if len(X_train_init) != len(sensitive_train_init):
            raise ValueError(
                "X_train_init and sensitive_train_init length mismatch"
            )

        rng = np.random.default_rng(self.random_state)

        # The growing training pool. We hold X, y, and the primary
        # sensitive series in lock-step so feedback updates are coherent.
        X_pool = X_train_init.reset_index(drop=True).copy()
        y_pool = y_train_init.reset_index(drop=True).copy()
        s_pool = sensitive_train_init.reset_index(drop=True).copy()

        applicants_n = (
            self.applicants_per_round
            if self.applicants_per_round is not None
            else len(X_train_init)
        )

        results: list[RoundResult] = []
        for round_idx in range(self.n_rounds):
            model = self.model_factory()
            model.train(X_pool, y_pool, sensitive_features=s_pool)

            test_preds = np.asarray(model.predict(X_test))
            metrics = self._round_metrics(y_test, test_preds, sensitive_test)
            per_sensitive = {
                self.sensitive_features[0]: metrics
            }

            # Draw the next round's applicant pool from the *initial*
            # training distribution. This is a stylised choice; see the
            # RFC for why we do not let the applicant pool drift on its
            # own.
            n_init = len(X_train_init)
            applicant_idx = rng.integers(0, n_init, size=applicants_n)
            X_app = (
                X_train_init.reset_index(drop=True)
                .iloc[applicant_idx]
                .reset_index(drop=True)
            )
            y_app = (
                y_train_init.reset_index(drop=True)
                .iloc[applicant_idx]
                .reset_index(drop=True)
            )
            s_app = (
                sensitive_train_init.reset_index(drop=True)
                .iloc[applicant_idx]
                .reset_index(drop=True)
            )

            scores = self._get_scores(model, X_app)
            selected = apply_decision_rule(
                scores,
                self.decision_rule,
                threshold=self.threshold,
                k=self.top_k,
            )

            selected_mask = np.zeros(len(X_app), dtype=bool)
            selected_mask[selected] = True
            per_group_rate = self._per_group_selection_rate(s_app, selected_mask)

            results.append(
                RoundResult(
                    round_idx=round_idx,
                    metrics=metrics,
                    per_sensitive=per_sensitive,
                    n_train=len(X_pool),
                    n_selected=int(len(selected)),
                    selection_rate_per_group=per_group_rate,
                )
            )

            # Feedback: update the pool for the next round.
            X_new, y_new, s_new = apply_feedback(
                self.feedback_rule, X_app, y_app, s_app, selected
            )
            if len(X_new) > 0:
                X_pool = pd.concat([X_pool, X_new], ignore_index=True)
                y_pool = pd.concat([y_pool, y_new], ignore_index=True)
                s_pool = pd.concat([s_pool, s_new], ignore_index=True)

        return results

    # -- summary ------------------------------------------------------------

    @staticmethod
    def summary(results: list[RoundResult]) -> dict[str, Any]:
        """Aggregate a run into headline stats: mean, stddev, slope of DP."""
        if not results:
            return {
                "n_rounds": 0,
                "dp_mean": float("nan"),
                "dp_std": float("nan"),
                "eo_mean": float("nan"),
                "eo_std": float("nan"),
                "accuracy_mean": float("nan"),
                "dp_slope": float("nan"),
                "dp_first": float("nan"),
                "dp_last": float("nan"),
                "drift_headline": "no rounds recorded",
            }
        dp = np.array(
            [abs(r.metrics.get("demographic_parity_difference", 0.0)) for r in results]
        )
        eo = np.array(
            [
                abs(r.metrics.get("equalized_odds_difference", 0.0))
                if not np.isnan(r.metrics.get("equalized_odds_difference", 0.0))
                else 0.0
                for r in results
            ]
        )
        acc = np.array([r.metrics.get("accuracy", 0.0) for r in results])

        rounds = np.arange(len(results), dtype=float)
        if len(results) >= 2:
            slope, _intercept = np.polyfit(rounds, dp, 1)
        else:
            slope = 0.0

        n = len(results)
        headline = (
            f"DP gap moved from {dp[0]:.3f} at round 0 to {dp[-1]:.3f} at "
            f"round {n - 1} ({n} rounds, slope {slope:+.4f}/round)"
        )

        return {
            "n_rounds": n,
            "dp_mean": float(dp.mean()),
            "dp_std": float(dp.std(ddof=0)),
            "eo_mean": float(eo.mean()),
            "eo_std": float(eo.std(ddof=0)),
            "accuracy_mean": float(acc.mean()),
            "dp_slope": float(slope),
            "dp_first": float(dp[0]),
            "dp_last": float(dp[-1]),
            "drift_headline": headline,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_result_to_dict(r: RoundResult) -> dict[str, Any]:
    """Convenience wrapper around dataclasses.asdict for one RoundResult."""
    return asdict(r)
