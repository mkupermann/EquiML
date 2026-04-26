"""Group-uncertainty aware fairness intervals.

See docs/rfcs/0006-group-uncertainty-fairness.md.

Real protected attributes are not zero-uncertainty. They are sampled from a
finite test set, missing for some rows, and noisy when present. Reporting
fairness metrics as point estimates hides all three sources of uncertainty.

This module produces interval estimates by composing three simulators:

1. Bootstrap          — sample uncertainty from the finite test set.
2. Multiple imputation — uncertainty introduced by missing protected attrs.
3. Noise-matrix       — uncertainty from a known label-noise process on the
                        observed protected attribute.

Each simulator emits a population of fairness-metric draws; the interval is
the percentile band of that population at the requested confidence level.

Limitations (named honestly, not buried):

- Bootstrap assumes the test sample is representative of the deployment
  distribution. If the test set is biased relative to deployment, the
  interval is centred on the wrong number.

- Multiple imputation here assumes data is **missing at random (MAR)**
  conditional on the observed features passed in (or **missing completely
  at random (MCAR)** if no features are passed). If protected attributes
  are missing for reasons correlated with the outcome (NMAR), the
  imputation interval is optimistic — it will look tight while being
  systematically off.

- The noise model assumes the user supplies a credible
  P(observed | true) matrix. Garbage in, garbage out. We do not estimate
  the noise matrix from data; that needs auxiliary ground truth which by
  definition we do not have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)

logger = logging.getLogger(__name__)


# Metric keys used throughout. Kept in one place so the dataclass and the
# simulators stay in lockstep.
DP = "demographic_parity_difference"
EO = "equalized_odds_difference"
EOPP = "equal_opportunity_difference"
DEFAULT_METRICS: tuple[str, ...] = (DP, EO, EOPP)


# --- public dataclasses ----------------------------------------------------


@dataclass
class MetricInterval:
    """A fairness metric reported with a confidence interval.

    `point` is the estimate on the input data as-is (no resampling).
    `lower` and `upper` are the percentile-band edges across the simulator
    population. `confidence` is the nominal coverage (e.g. 0.95).
    """

    point: float
    lower: float
    upper: float
    confidence: float = 0.95

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return f"{self.point:.2f} [{self.lower:.2f}, {self.upper:.2f}] ({pct}%)"

    def width(self) -> float:
        return float(self.upper - self.lower)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass
class UncertaintyResult:
    """Aggregated interval estimates across DP / EO / EOpp."""

    metrics: dict[str, MetricInterval]
    n_bootstrap: int
    n_imputations: int
    missing_fraction: float
    notes: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> MetricInterval:
        return self.metrics[key]

    def keys(self) -> Iterable[str]:
        return self.metrics.keys()


# --- internal helpers ------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                "sensitive_features as DataFrame must have exactly one column; "
                "pass a Series for the single sensitive attribute."
            )
        return x.iloc[:, 0].to_numpy()
    return np.asarray(x)


def _check_inputs(y_true, y_pred, sensitive) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    sf = _to_numpy(sensitive)
    if len(yt) == 0 or len(yp) == 0 or len(sf) == 0:
        raise ValueError("Inputs must be non-empty.")
    if not (len(yt) == len(yp) == len(sf)):
        raise ValueError(
            f"Length mismatch: y_true={len(yt)}, y_pred={len(yp)}, "
            f"sensitive_features={len(sf)}."
        )
    return yt, yp, sf


def _safe_metric(fn, y_true, y_pred, sensitive_features) -> float:
    """Compute a fairlearn fairness metric, returning 0.0 on degenerate input.

    Degenerate cases that would otherwise raise:
      - only one group present in the resample
      - no positives in y_true (kills EOpp)
      - constant predictions (DP is well-defined; EO/EOpp may degrade)
    """
    try:
        # If only one group is present, fairness gaps are undefined; we
        # report 0.0 (no observable disparity given the resample).
        if len(np.unique(sensitive_features)) < 2:
            return 0.0
        return float(fn(y_true, y_pred, sensitive_features=sensitive_features))
    except (ValueError, ZeroDivisionError) as exc:
        logger.debug("Fairness metric degraded to 0.0: %s", exc)
        return 0.0


def _point_estimates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sf: np.ndarray,
) -> dict[str, float]:
    return {
        DP: _safe_metric(demographic_parity_difference, y_true, y_pred, sf),
        EO: _safe_metric(equalized_odds_difference, y_true, y_pred, sf),
        EOPP: _safe_metric(equal_opportunity_difference, y_true, y_pred, sf),
    }


def _percentile_intervals(
    samples: dict[str, list[float]],
    point: dict[str, float],
    confidence: float,
) -> dict[str, MetricInterval]:
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1).")
    alpha = 1.0 - confidence
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out: dict[str, MetricInterval] = {}
    for k, draws in samples.items():
        arr = np.asarray(draws, dtype=float)
        if arr.size == 0:
            out[k] = MetricInterval(
                point=point[k], lower=point[k], upper=point[k],
                confidence=confidence,
            )
            continue
        out[k] = MetricInterval(
            point=point[k],
            lower=float(np.percentile(arr, lo_q)),
            upper=float(np.percentile(arr, hi_q)),
            confidence=confidence,
        )
    return out


# --- 1. bootstrap ----------------------------------------------------------


def bootstrap_fairness_intervals(
    y_true,
    y_pred,
    sensitive_features,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = 42,
) -> UncertaintyResult:
    """Non-parametric bootstrap for DP / EO / EOpp on a fully-observed sample.

    Resamples rows with replacement `n_bootstrap` times, recomputes each
    fairness metric per resample, and reports the empirical percentile band.

    The interval reflects sample uncertainty only — i.e. the variance from
    seeing one finite test set rather than the population. It does NOT
    account for missingness or label noise in the sensitive feature; for
    those, see `impute_and_bootstrap` and `noise_model_intervals`.

    Limitation: bootstrap assumes the test sample is representative of the
    deployment distribution. If it is not, the interval is centred on the
    wrong number.
    """
    yt, yp, sf = _check_inputs(y_true, y_pred, sensitive_features)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")

    point = _point_estimates(yt, yp, sf)
    samples: dict[str, list[float]] = {DP: [], EO: [], EOPP: []}

    rng = np.random.default_rng(random_state)
    n = len(yt)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s = sf[idx]
        # Skip resamples with only one group present rather than poisoning
        # the population with synthetic 0.0s.
        if len(np.unique(s)) < 2:
            continue
        yti, ypi = yt[idx], yp[idx]
        samples[DP].append(_safe_metric(demographic_parity_difference, yti, ypi, s))
        samples[EO].append(_safe_metric(equalized_odds_difference, yti, ypi, s))
        samples[EOPP].append(_safe_metric(equal_opportunity_difference, yti, ypi, s))

    intervals = _percentile_intervals(samples, point, confidence)
    notes: list[str] = []
    if any(len(v) < n_bootstrap for v in samples.values()):
        notes.append(
            "Some bootstrap resamples contained only one sensitive group "
            "and were skipped."
        )
    return UncertaintyResult(
        metrics=intervals,
        n_bootstrap=n_bootstrap,
        n_imputations=0,
        missing_fraction=0.0,
        notes=notes,
    )


# --- 2. multiple imputation -----------------------------------------------


def _empirical_distribution(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (categories, probabilities) for non-NaN entries."""
    obs = values[~pd.isna(values)]
    if obs.size == 0:
        raise ValueError("All sensitive values are missing; cannot impute.")
    cats, counts = np.unique(obs, return_counts=True)
    return cats, counts / counts.sum()


def _impute_marginal(
    sf_with_nan: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fill NaN positions by drawing from the marginal empirical distribution."""
    out = sf_with_nan.copy()
    mask = pd.isna(out)
    if not mask.any():
        return out
    cats, probs = _empirical_distribution(out)
    out[mask] = rng.choice(cats, size=int(mask.sum()), p=probs)
    return out


def _impute_conditional(
    sf_with_nan: np.ndarray,
    features: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fill NaN positions by drawing from the empirical distribution
    conditional on a coarse binning of the supplied features.

    Bins each numeric feature into quartiles (categorical features used
    as-is) and forms a composite stratum key. For each stratum, draws
    imputations from the within-stratum empirical distribution. Falls back
    to the marginal distribution for strata where every row is missing.
    """
    out = sf_with_nan.copy()
    mask = pd.isna(out)
    if not mask.any():
        return out

    if features is None or features.shape[1] == 0:
        return _impute_marginal(out, rng)

    # Build a stratum key per row.
    strata = pd.DataFrame(index=features.index)
    for col in features.columns:
        s = features[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                strata[col] = pd.qcut(s, q=4, duplicates="drop", labels=False)
            except ValueError:
                strata[col] = 0
        else:
            strata[col] = s.astype(str)
    key = strata.astype(str).agg("|".join, axis=1).to_numpy()

    cats_global, probs_global = _empirical_distribution(out)
    for stratum in np.unique(key):
        sel = (key == stratum)
        sel_obs = sel & ~mask
        sel_miss = sel & mask
        if not sel_miss.any():
            continue
        if sel_obs.any():
            cats, probs = _empirical_distribution(out[sel_obs])
        else:
            cats, probs = cats_global, probs_global
        out[sel_miss] = rng.choice(cats, size=int(sel_miss.sum()), p=probs)
    return out


def impute_and_bootstrap(
    y_true,
    y_pred,
    sensitive_features_with_nan,
    features: pd.DataFrame | None = None,
    n_imputations: int = 20,
    n_bootstrap_per_impute: int = 200,
    confidence: float = 0.95,
    random_state: int | None = 42,
) -> UncertaintyResult:
    """Multiple-imputation + bootstrap fairness intervals for partly-missing
    protected attributes.

    For each of `n_imputations` imputations:
      1. Fill NaN sensitive values by sampling from the empirical
         distribution (marginal, or conditional on `features` if supplied).
      2. Bootstrap fairness metrics on the imputed dataset
         (`n_bootstrap_per_impute` resamples).

    The bootstrap draws are pooled across imputations and the percentile
    band is taken on the pooled population. This is a simpler alternative
    to Rubin's rules; for the audit-context use case (variability bands,
    not p-values) it is well-behaved and avoids the hidden assumptions
    Rubin's rules carry on the within-imputation variance estimator.

    Limitation: imputation here assumes the protected attribute is missing
    at random conditional on `features` (MAR), or completely at random
    (MCAR) if no features are passed. If missingness is correlated with
    the outcome (NMAR), this interval is optimistic.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    sf = _to_numpy(sensitive_features_with_nan)

    if len(yt) == 0 or len(yp) == 0 or len(sf) == 0:
        raise ValueError("Inputs must be non-empty.")
    if not (len(yt) == len(yp) == len(sf)):
        raise ValueError("Length mismatch in y_true / y_pred / sensitive_features.")
    if n_imputations < 1:
        raise ValueError("n_imputations must be >= 1.")
    if n_bootstrap_per_impute < 1:
        raise ValueError("n_bootstrap_per_impute must be >= 1.")

    missing_mask = pd.isna(sf)
    missing_fraction = float(missing_mask.mean())
    if missing_fraction == 0:
        # No missingness — defer to the standard bootstrap.
        result = bootstrap_fairness_intervals(
            yt, yp, sf,
            n_bootstrap=n_imputations * n_bootstrap_per_impute,
            confidence=confidence,
            random_state=random_state,
        )
        result.notes.insert(
            0,
            "No missing sensitive values; impute_and_bootstrap delegated to plain bootstrap.",
        )
        return result

    # Point estimate: imputed once with a fixed seed, so the headline
    # number is reproducible. The interval below averages over many
    # imputations.
    rng_point = np.random.default_rng(random_state)
    sf_point = (
        _impute_conditional(sf, features, rng_point)
        if features is not None and features.shape[1] > 0
        else _impute_marginal(sf, rng_point)
    )
    point = _point_estimates(yt, yp, sf_point)

    samples: dict[str, list[float]] = {DP: [], EO: [], EOPP: []}
    rng = np.random.default_rng(random_state)
    n = len(yt)
    skipped = 0
    for _ in range(n_imputations):
        if features is not None and features.shape[1] > 0:
            sf_imp = _impute_conditional(sf, features, rng)
        else:
            sf_imp = _impute_marginal(sf, rng)
        for _ in range(n_bootstrap_per_impute):
            idx = rng.integers(0, n, size=n)
            s = sf_imp[idx]
            if len(np.unique(s)) < 2:
                skipped += 1
                continue
            yti, ypi = yt[idx], yp[idx]
            samples[DP].append(_safe_metric(demographic_parity_difference, yti, ypi, s))
            samples[EO].append(_safe_metric(equalized_odds_difference, yti, ypi, s))
            samples[EOPP].append(_safe_metric(equal_opportunity_difference, yti, ypi, s))

    intervals = _percentile_intervals(samples, point, confidence)
    notes = [
        f"Multiple imputation: {n_imputations} imputations × "
        f"{n_bootstrap_per_impute} bootstrap resamples each.",
        "Imputation assumes MAR given features (or MCAR if no features).",
    ]
    if skipped:
        notes.append(
            f"{skipped} resample(s) skipped because only one sensitive group was present."
        )
    return UncertaintyResult(
        metrics=intervals,
        n_bootstrap=n_imputations * n_bootstrap_per_impute,
        n_imputations=n_imputations,
        missing_fraction=missing_fraction,
        notes=notes,
    )


# --- 3. noise model --------------------------------------------------------


def noise_model_intervals(
    y_true,
    y_pred,
    sensitive_observed,
    error_rate_matrix: pd.DataFrame | dict | np.ndarray,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = 42,
) -> UncertaintyResult:
    """Fairness intervals under a known label-noise model on the observed
    protected attribute.

    `error_rate_matrix` encodes P(observed_label | true_label) as either:

    - a `pd.DataFrame` with rows = true labels, columns = observed labels,
      cells = conditional probabilities (each row sums to 1);
    - a `dict[true_label][observed_label] = probability`; or
    - a 2-D numpy array (rows = true, columns = observed) along with the
      label order taken from `np.unique(sensitive_observed)`.

    For each of `n_simulations`, we draw a `true_label` for each row by
    inverting the noise model (Bayes' rule with the empirical marginal of
    the observed labels as the prior on `true_label`), then compute
    fairness metrics against the simulated true labels.

    Limitation: the user supplies the noise matrix; we do not estimate it.
    A wrong matrix produces a confidently wrong interval.
    """
    yt, yp, so = _check_inputs(y_true, y_pred, sensitive_observed)
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1.")

    labels = np.unique(so)
    matrix = _normalise_noise_matrix(error_rate_matrix, labels)

    # Prior on true labels: take the empirical observed marginal as a
    # crude prior (better than uniform, doesn't need extra arguments).
    obs_counts = np.array([(so == lab).sum() for lab in labels], dtype=float)
    prior = obs_counts / obs_counts.sum()

    # Posterior P(true | observed) per row, via Bayes.
    # P(true=t | obs=o) ∝ P(obs=o | true=t) * P(true=t)
    # matrix[t, o] = P(obs=o | true=t).
    # We need, for each observed label o, a vector over t.
    posteriors: dict = {}
    for j, o in enumerate(labels):
        unnorm = matrix[:, j] * prior
        z = unnorm.sum()
        posteriors[o] = (unnorm / z) if z > 0 else prior

    # Point estimate: use the observed labels as-is.
    point = _point_estimates(yt, yp, so)

    samples: dict[str, list[float]] = {DP: [], EO: [], EOPP: []}
    rng = np.random.default_rng(random_state)
    n = len(yt)
    skipped = 0
    for _ in range(n_simulations):
        # Per-row draw of true label from its posterior.
        drawn = np.empty(n, dtype=labels.dtype)
        for o in labels:
            mask = (so == o)
            k = int(mask.sum())
            if k == 0:
                continue
            drawn[mask] = rng.choice(labels, size=k, p=posteriors[o])
        if len(np.unique(drawn)) < 2:
            skipped += 1
            continue
        samples[DP].append(_safe_metric(demographic_parity_difference, yt, yp, drawn))
        samples[EO].append(_safe_metric(equalized_odds_difference, yt, yp, drawn))
        samples[EOPP].append(_safe_metric(equal_opportunity_difference, yt, yp, drawn))

    intervals = _percentile_intervals(samples, point, confidence)
    notes = [
        f"Noise-model simulation: {n_simulations} draws of the true sensitive label.",
        "Assumes the supplied error_rate_matrix is correct; we do not validate it.",
    ]
    if skipped:
        notes.append(
            f"{skipped} simulation(s) collapsed to a single group and were skipped."
        )
    return UncertaintyResult(
        metrics=intervals,
        n_bootstrap=n_simulations,
        n_imputations=0,
        missing_fraction=0.0,
        notes=notes,
    )


def _normalise_noise_matrix(
    matrix: pd.DataFrame | dict | np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Coerce a noise specification to a square row-stochastic ndarray."""
    k = len(labels)
    if isinstance(matrix, pd.DataFrame):
        m = matrix.reindex(index=labels, columns=labels).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(matrix, dict):
        m = np.zeros((k, k), dtype=float)
        for i, t in enumerate(labels):
            row = matrix.get(t, {})
            for j, o in enumerate(labels):
                m[i, j] = float(row.get(o, 0.0))
    else:
        m = np.asarray(matrix, dtype=float)
        if m.shape != (k, k):
            raise ValueError(
                f"error_rate_matrix shape {m.shape} does not match number "
                f"of observed labels ({k})."
            )

    # Row-normalise. A row that is all-zero is replaced by the identity
    # (no-noise) row; this keeps the simulator stable when the user
    # forgets to specify a row.
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    m = m / row_sums
    for i in range(k):
        if m[i].sum() == 0:
            m[i, i] = 1.0
    return m


__all__ = [
    "DP",
    "EO",
    "EOPP",
    "DEFAULT_METRICS",
    "MetricInterval",
    "UncertaintyResult",
    "bootstrap_fairness_intervals",
    "impute_and_bootstrap",
    "noise_model_intervals",
]
