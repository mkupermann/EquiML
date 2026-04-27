"""Naive counterfactual fairness and proxy detection.

See docs/rfcs/0004-counterfactual-fairness.md.

This module ships a *sensitivity-style* counterfactual audit, not a causal
counterfactual in the Pearl/Kusner sense. We do not assume a structural
causal model. We flip the protected attribute on the test set, re-predict,
and report the share of predictions that change. That number is a useful
diagnostic. It is not a causal claim.

Two functions are exported:

- ``compute_counterfactual_audit`` — flips the protected attribute on the
  test set and reports the flip rate (and, if ``predict_proba`` is available,
  the mean shift in P(y=1)).
- ``compute_proxy_features`` — for each candidate non-protected feature,
  compares the flip rate when the feature is included against the flip rate
  when the feature is neutralised (set to its global mean for numerical,
  global mode for categorical). Features whose neutralisation makes the
  flip rate go *up* are carrying / masking the protected-attribute signal:
  removing them forces the model to lean directly on the protected
  attribute, which is what the flip-test exposes. We call those proxies
  and rank them by ``proxy_strength = flip_rate_without - flip_rate_with``.

The honest framing: this is sensitivity analysis. A flip rate of 0.30 says
"30% of predictions would change if we toggled the protected attribute,
holding other features as recorded." Whether that is acceptable, lawful,
or causal is for the human reviewer (and possibly a domain causal graph)
to decide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProxyFeature:
    """One non-protected feature ranked by how much it carries the protected signal.

    Fields:

    - ``flip_rate_with`` — counterfactual flip rate on the original
      ``X_test`` (the candidate feature is left as observed).
    - ``flip_rate_without`` — flip rate on a copy of ``X_test`` where the
      candidate feature has been set to its global neutral value (mean for
      numerical, mode for categorical).
    - ``proxy_strength`` — ``flip_rate_without - flip_rate_with``. Positive
      values mean that neutralising the candidate feature *increased* the
      model's protected-attribute sensitivity, i.e. the candidate was
      previously masking / carrying the protected signal. Larger ⇒ more
      proxy-like. Values near zero or negative mean the candidate was not
      a proxy.
    """
    feature_name: str
    flip_rate_with: float
    flip_rate_without: float
    proxy_strength: float


@dataclass
class CounterfactualResult:
    """Result of a naive counterfactual audit.

    Attributes
    ----------
    flip_rate
        Share of test-set predictions that change when the protected
        attribute is flipped (binary) or cycled through other levels and
        averaged (multi-class).
    mean_prediction_shift
        For models with ``predict_proba``: the mean of
        ``|p_flipped - p_original|`` for the positive class. ``None`` if the
        model exposes no probability output (e.g. fairness-mitigated
        models without a well-defined ``predict_proba``).
    proxy_features
        Empty by default. Populated by ``compute_proxy_features`` when the
        caller asks for proxy ranking.
    n_samples
        Number of test-set rows audited.
    notes
        Free-form caveats produced during the run (e.g. a sensitive feature
        with only one observed level).
    """
    flip_rate: float
    mean_prediction_shift: float | None
    proxy_features: list[ProxyFeature] = field(default_factory=list)
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_sensitive(
    X_test: pd.DataFrame,
    sensitive_feature: str | pd.Series,
) -> tuple[str | None, pd.Series]:
    """Return (column_name, series) for the protected attribute.

    Accepts either a column name present in ``X_test`` or a standalone
    ``pd.Series`` (in which case the returned column name is ``None``).
    """
    if isinstance(sensitive_feature, pd.Series):
        if len(sensitive_feature) != len(X_test):
            raise ValueError(
                "sensitive_feature Series length does not match X_test."
            )
        return sensitive_feature.name, sensitive_feature.reset_index(drop=True)

    if not isinstance(sensitive_feature, str):
        raise TypeError(
            "sensitive_feature must be a column name (str) or pd.Series."
        )
    if sensitive_feature not in X_test.columns:
        raise KeyError(
            f"sensitive_feature {sensitive_feature!r} not in X_test columns."
        )
    return sensitive_feature, X_test[sensitive_feature].reset_index(drop=True)


def _infer_sensitive_values(
    series: pd.Series,
    sensitive_values: Sequence | None,
) -> list:
    """Return the ordered list of levels to cycle through.

    If the caller did not pass ``sensitive_values``, use the unique values
    observed in the series (sorted for determinism).
    """
    if sensitive_values is not None:
        levels = list(sensitive_values)
    else:
        levels = sorted(series.dropna().unique().tolist())
    if len(levels) < 2:
        raise ValueError(
            "Counterfactual audit requires at least two levels for the "
            f"protected attribute; got {levels!r}."
        )
    return levels


def _swap_or_cycle(
    X: pd.DataFrame,
    column: str,
    original: pd.Series,
    levels: list,
) -> Iterable[pd.DataFrame]:
    """Yield counterfactual frames where every row has its protected level shifted.

    For binary protected attributes (``len(levels) == 2``) yields one frame
    in which every row is set to "the other level". For multi-class, yields
    one frame per non-original level so the caller can average flip rates
    across all alternatives.
    """
    if len(levels) == 2:
        a, b = levels
        flipped = X.copy()
        flipped[column] = original.map(lambda v: b if v == a else a)
        yield flipped
        return

    for target in levels:
        # For each non-original target, build a frame that sets every row
        # whose original level differs from `target` to `target`. Rows
        # whose original IS `target` are left alone (they would be a no-op
        # flip and contribute nothing to the flip rate denominator we use
        # below).
        flipped = X.copy()
        flipped[column] = np.where(original == target, original, target)
        yield flipped


def _has_proba(model: Any) -> bool:
    """Return True if ``model.predict_proba`` is callable and well-defined.

    The EquiML ``Model`` class raises ``NotImplementedError`` for the
    fairness-mitigated estimator (ExponentiatedGradient) on purpose; we
    treat that as "no probability output" rather than crashing the audit.
    """
    if not hasattr(model, "predict_proba"):
        return False
    try:
        # Probe with an empty-shape call by inspecting the underlying
        # estimator instead of actually calling. The ``Model`` wrapper
        # exposes ``predictors_`` on the mitigated path.
        if hasattr(model, "predictors_"):
            return False
        if hasattr(model, "model") and hasattr(model.model, "predictors_"):
            return False
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _safe_proba_positive(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Return P(y=1) for ``X`` if the model supports it; otherwise ``None``."""
    if not _has_proba(model):
        return None
    try:
        proba = model.predict_proba(X)
    except (NotImplementedError, AttributeError):
        return None
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] < 2:
        return None
    # Convention: positive class is the last column. sklearn binary
    # classifiers expose classes_ in sorted order, so column 1 is class 1.
    return proba[:, -1]


def _neutralised_value(series: pd.Series) -> Any:
    """Return the global "neutral" value for a feature.

    Numerical: the mean. Categorical / object: the mode (first if ties).
    Used by ``compute_proxy_features`` to remove the feature's signal
    without dropping the column (which would change the model's expected
    input shape).
    """
    if pd.api.types.is_numeric_dtype(series):
        return float(series.mean())
    mode = series.mode(dropna=True)
    if len(mode) == 0:
        # Series is entirely NaN; neutralise to NaN.
        return np.nan
    return mode.iloc[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_counterfactual_audit(
    model: Any,
    X_test: pd.DataFrame,
    sensitive_feature: str | pd.Series,
    sensitive_values: Sequence | None = None,
) -> CounterfactualResult:
    """Run a naive counterfactual audit on ``model``.

    For each row in ``X_test``, flip the protected attribute (binary) or
    cycle through every other level (multi-class), re-predict, and report
    the share of predictions that change.

    Parameters
    ----------
    model
        Anything with a ``predict(X)`` method. ``predict_proba(X)`` is used
        when available to compute the mean shift in P(y=1); models that do
        not support a well-defined probability output (the fairness-
        mitigated ``ExponentiatedGradient`` path) are handled gracefully.
    X_test
        The test-set features. The protected attribute must already be
        present as a column unless ``sensitive_feature`` is passed as a
        standalone ``pd.Series``.
    sensitive_feature
        Either a column name in ``X_test`` or a standalone ``pd.Series`` of
        the same length. The Series form is useful when the model's
        feature matrix has been one-hot encoded and the protected attribute
        no longer exists as a single column.
    sensitive_values
        Optional list of levels to use. If not provided, uses the unique
        values observed in the protected column. Useful when the test set
        does not contain every level the model was trained on.

    Returns
    -------
    CounterfactualResult
        See the dataclass for fields. ``proxy_features`` is empty here;
        use ``compute_proxy_features`` to populate it.

    Notes
    -----
    This is sensitivity analysis, not causal counterfactual fairness in the
    Pearl/Kusner sense. A high flip rate means the model is highly
    sensitive to the protected attribute as a feature; whether that
    sensitivity is causal, lawful, or acceptable is a separate question.
    """
    column, original_series = _resolve_sensitive(X_test, sensitive_feature)
    notes: list[str] = []

    if column is None:
        # Standalone Series; nothing to flip in X_test directly.
        # We still need a column name to write the flipped value to.
        raise ValueError(
            "compute_counterfactual_audit currently requires the protected "
            "attribute to live in X_test as a named column. Pass the column "
            "name (str) rather than a standalone Series."
        )

    levels = _infer_sensitive_values(original_series, sensitive_values)

    X = X_test.reset_index(drop=True).copy()
    original_series = original_series.reset_index(drop=True)

    base_pred = np.asarray(model.predict(X))
    base_proba_pos = _safe_proba_positive(model, X)

    flip_rates: list[float] = []
    proba_shifts: list[float] = []

    for flipped_X in _swap_or_cycle(X, column, original_series, levels):
        flipped_pred = np.asarray(model.predict(flipped_X))
        flip_rates.append(float(np.mean(flipped_pred != base_pred)))

        if base_proba_pos is not None:
            flipped_proba = _safe_proba_positive(model, flipped_X)
            if flipped_proba is not None:
                proba_shifts.append(
                    float(np.mean(np.abs(flipped_proba - base_proba_pos)))
                )

    flip_rate = float(np.mean(flip_rates)) if flip_rates else 0.0
    mean_shift: float | None = (
        float(np.mean(proba_shifts)) if proba_shifts else None
    )
    if base_proba_pos is None:
        notes.append(
            "Model exposes no well-defined predict_proba; mean_prediction_shift omitted."
        )

    if len(levels) > 2:
        notes.append(
            f"Multi-class protected attribute with levels {levels!r}; "
            "flip rate is averaged across each non-original target."
        )

    return CounterfactualResult(
        flip_rate=flip_rate,
        mean_prediction_shift=mean_shift,
        proxy_features=[],
        n_samples=int(len(X)),
        notes=notes,
    )


def compute_proxy_features(
    model: Any,
    X_test: pd.DataFrame,
    sensitive_feature: str | pd.Series,
    candidate_features: list[str] | None = None,
    sensitive_values: Sequence | None = None,
    top_k: int = 10,
) -> list[ProxyFeature]:
    """Rank non-protected features by how much they carry the protected signal.

    For each candidate feature, compute the counterfactual flip rate twice:
    once on the original ``X_test`` and once on a copy in which the
    candidate feature is set to its global "neutral" value (mean for
    numerical, mode for categorical). Features whose neutralisation makes
    the flip rate go *up* are proxies: with the candidate present the
    model could lean on it instead of the protected attribute, masking the
    direct dependence; with the candidate neutralised that pathway is
    closed and flipping the protected attribute now flips more
    predictions.

    Parameters
    ----------
    model
        Anything with ``predict(X)``.
    X_test
        Test-set features.
    sensitive_feature
        Column name of the protected attribute in ``X_test``.
    candidate_features
        Non-protected columns to test. Defaults to every non-protected
        column in ``X_test``.
    sensitive_values
        Forwarded to ``compute_counterfactual_audit``.
    top_k
        Return only the top-k proxies, sorted by descending
        ``proxy_strength``.

    Returns
    -------
    list[ProxyFeature]
        Sorted by descending ``proxy_strength``. A positive value means
        neutralising the candidate raised the protected-attribute flip
        rate, indicating the feature was carrying the protected signal. A
        value near zero (or negative) means removing the feature did not
        amplify the model's protected-attribute sensitivity.
    """
    column, _ = _resolve_sensitive(X_test, sensitive_feature)
    if column is None:
        raise ValueError(
            "compute_proxy_features requires sensitive_feature to be a "
            "column name in X_test."
        )

    if candidate_features is None:
        candidate_features = [c for c in X_test.columns if c != column]
    else:
        unknown = [c for c in candidate_features if c not in X_test.columns]
        if unknown:
            raise KeyError(f"candidate_features not in X_test: {unknown!r}")
        candidate_features = [c for c in candidate_features if c != column]

    base = compute_counterfactual_audit(
        model, X_test, column, sensitive_values=sensitive_values
    )
    base_rate = base.flip_rate

    ranked: list[ProxyFeature] = []
    for feat in candidate_features:
        neutralised = X_test.copy()
        neutralised[feat] = _neutralised_value(X_test[feat])
        without = compute_counterfactual_audit(
            model, neutralised, column, sensitive_values=sensitive_values
        )
        ranked.append(
            ProxyFeature(
                feature_name=feat,
                flip_rate_with=base_rate,
                flip_rate_without=without.flip_rate,
                proxy_strength=without.flip_rate - base_rate,
            )
        )

    ranked.sort(key=lambda p: p.proxy_strength, reverse=True)
    if top_k is not None and top_k > 0:
        ranked = ranked[:top_k]
    return ranked


__all__ = [
    "CounterfactualResult",
    "ProxyFeature",
    "compute_counterfactual_audit",
    "compute_proxy_features",
]
