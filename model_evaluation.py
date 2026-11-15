"""
Purged walk-forward evaluation, time-aware probability calibration, and held-out explanations.

The chronological holdout in `train_model` stays a single-split diagnostic.
Trading probabilities in the technical path come from this module: each test
window is scored by a model that has not seen that window, and any calibrator
is fit only on an earlier slice of the same training window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model import ModelConfig, _build_classifier, _resolve_feature_cols
from utils import set_seed


MIN_FIT_ROWS = 40
MIN_TEST_ROWS = 15
MIN_SIGMOID_ROWS = 25
MIN_ISOTONIC_ROWS = 80
MIN_ISOTONIC_PER_CLASS = 8
MIN_IMPORTANCE_ROWS = 20
CALIBRATION_METHODS = ("auto", "sigmoid", "isotonic", "none")


@dataclass(frozen=True)
class FoldPlan:
    """Labeled-row bounds. Ends are exclusive. Fit, calibration, and test do not overlap."""

    fit_end: int
    calib_start: int | None
    calib_end: int | None
    test_start: int
    test_end: int
    embargo_bars: int


@dataclass
class WalkForwardEvaluation:
    folds: pd.DataFrame
    oos_frame: pd.DataFrame
    tail_frame: pd.DataFrame
    oos_y: np.ndarray
    oos_raw: np.ndarray
    oos_decision: np.ndarray
    oos_metrics: dict
    oos_raw_metrics: dict
    oos_sigmoid_metrics: dict | None
    oos_isotonic_metrics: dict | None
    insample_accuracy: float
    insample_n: int
    calibration_window_metrics: dict | None
    reliability_bins: pd.DataFrame
    reliability_bins_raw: pd.DataFrame
    confusion: pd.DataFrame
    importance: pd.DataFrame
    local_explanation: pd.DataFrame
    calibration_requested: str
    decision_method: str
    explained_date: pd.Timestamp | None
    explained_probability: float
    explained_position: int
    last_fit_end_date: pd.Timestamp | None


def _normalize_method(method: str) -> str:
    name = str(method).lower().strip()
    if name == "raw":
        name = "none"
    if name not in CALIBRATION_METHODS:
        raise ValueError("calibration must be 'auto', 'sigmoid', 'isotonic', or 'none'")
    return name


def _split_calibration(
    train_limit: int,
    horizon: int,
    *,
    min_fit: int = MIN_FIT_ROWS,
    min_calib: int = MIN_SIGMOID_ROWS,
) -> tuple[int, int | None, int | None]:
    """
    Split the pre-test labeled window into fit and calibration.

    `train_limit` is exclusive. The calibration labels end at `train_limit`,
    which is already `horizon` labeled rows before the test window.
    """
    gap = max(0, int(horizon))
    if train_limit < min_fit + gap + min_calib:
        return train_limit, None, None
    max_calib = train_limit - gap - min_fit
    if max_calib < min_calib:
        return train_limit, None, None
    calib_len = max(min_calib, int(round(0.2 * train_limit)))
    calib_len = min(calib_len, max_calib)
    calib_end = int(train_limit)
    calib_start = calib_end - calib_len
    fit_end = calib_start - gap
    if fit_end < min_fit:
        return train_limit, None, None
    return int(fit_end), int(calib_start), int(calib_end)


def plan_walk_forward_folds(
    n_labeled: int,
    *,
    n_folds: int = 4,
    min_train_fraction: float = 0.5,
    horizon_bars: int = 1,
    reserve_calibration: bool = True,
    min_fit: int = MIN_FIT_ROWS,
    min_test: int = MIN_TEST_ROWS,
) -> list[FoldPlan]:
    """
    Expanding chronological folds.

    Test windows are contiguous and cover every labeled row after the minimum
    training prefix. Each training label ends at least `horizon_bars` rows
    before its test window. When calibration is reserved, the same embargo
    sits between the fit rows and the calibration rows.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not (0.2 <= float(min_train_fraction) < 0.95):
        raise ValueError("min_train_fraction must be in [0.2, 0.95)")
    horizon = max(0, int(horizon_bars))
    n_labeled = int(n_labeled)
    min_train = int(n_labeled * float(min_train_fraction))
    if min_train - horizon < min_fit:
        raise ValueError("Not enough rows for walk-forward validation")
    remaining = n_labeled - min_train
    if remaining < min_test:
        raise ValueError("Not enough rows for walk-forward validation")

    step = max(min_test, remaining // n_folds)
    bounds: list[tuple[int, int]] = []
    start = min_train
    while start < n_labeled and len(bounds) < n_folds:
        end = n_labeled if len(bounds) == n_folds - 1 else min(n_labeled, start + step)
        if n_labeled - end < min_test:
            end = n_labeled
        if end - start < min_test:
            if bounds:
                prev_start, _prev_end = bounds[-1]
                bounds[-1] = (prev_start, n_labeled)
            break
        bounds.append((start, end))
        if end == n_labeled:
            break
        start = end

    plans: list[FoldPlan] = []
    for test_start, test_end in bounds:
        train_limit = test_start - horizon
        if train_limit < min_fit:
            raise ValueError("Embargo leaves no training rows for a walk-forward fold")
        if reserve_calibration:
            fit_end, calib_start, calib_end = _split_calibration(train_limit, horizon, min_fit=min_fit)
        else:
            fit_end, calib_start, calib_end = train_limit, None, None
        plans.append(
            FoldPlan(
                fit_end=int(fit_end),
                calib_start=calib_start,
                calib_end=calib_end,
                test_start=int(test_start),
                test_end=int(test_end),
                embargo_bars=horizon,
            )
        )
    if not plans:
        raise ValueError("Failed to create walk-forward folds. Need more history.")
    return plans


def fit_probability_calibrator(raw_proba: np.ndarray, y: np.ndarray, method: str):
    """
    Fit one calibrator on an already time-ordered slice.

    Returns None when the slice is too small or has a single class. The caller
    then keeps the raw probability. `auto` is not resolved here: the walk-forward
    loop fits sigmoid and isotonic separately and picks with a size rule.
    """
    name = _normalize_method(method)
    if name == "auto":
        raise ValueError("fit_probability_calibrator expects 'sigmoid' or 'isotonic'")
    if name == "none":
        return None
    raw = np.asarray(raw_proba, dtype=float).ravel()
    labels = np.asarray(y, dtype=int).ravel()
    if len(raw) != len(labels):
        raise ValueError("raw_proba and y must have the same length")
    if not np.isfinite(raw).all():
        return None
    min_n = MIN_ISOTONIC_ROWS if name == "isotonic" else MIN_SIGMOID_ROWS
    if len(labels) < min_n or len(np.unique(labels)) < 2:
        return None
    if name == "isotonic":
        counts = np.bincount(labels, minlength=2)
        if int(counts.min()) < MIN_ISOTONIC_PER_CLASS:
            return None
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(raw, labels)
        return ("isotonic", model)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(raw.reshape(-1, 1), labels)
    return ("sigmoid", model)


def apply_probability_calibrator(calibrator, raw_proba: np.ndarray) -> np.ndarray:
    """Map raw probabilities. A missing calibrator returns the raw values unchanged."""
    raw = np.asarray(raw_proba, dtype=float).ravel()
    if calibrator is None:
        return raw.copy()
    kind, model = calibrator
    if kind == "isotonic":
        out = np.asarray(model.predict(raw), dtype=float)
    else:
        out = np.asarray(model.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
    return np.clip(out, 0.0, 1.0)


def _choose_decision(requested: str, sigmoid_cal, isotonic_cal):
    """
    Choose the decision calibrator without looking at the test fold.

    `auto` uses isotonic only when that fold's calibration slice was large
    enough to fit it, otherwise sigmoid, otherwise raw.
    """
    name = _normalize_method(requested)
    if name == "none":
        return None, "raw"
    if name == "sigmoid":
        return (sigmoid_cal, "sigmoid") if sigmoid_cal is not None else (None, "raw")
    if name == "isotonic":
        return (isotonic_cal, "isotonic") if isotonic_cal is not None else (None, "raw")
    if isotonic_cal is not None:
        return isotonic_cal, "isotonic"
    if sigmoid_cal is not None:
        return sigmoid_cal, "sigmoid"
    return None, "raw"


class _DecisionModel:
    """Base classifier plus an optional calibrator. Used only to score held-out rows."""

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        raw = np.asarray(self.base.predict_proba(np.asarray(X, dtype=float))[:, 1], dtype=float)
        p = apply_probability_calibrator(self.calibrator, raw)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def expected_calibration_error(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Weighted absolute gap between predicted probability and observed frequency."""
    labels = np.asarray(y, dtype=float).ravel()
    p = np.clip(np.asarray(proba, dtype=float).ravel(), 0.0, 1.0)
    if len(labels) == 0 or len(labels) != len(p):
        return float("nan"), pd.DataFrame(columns=["bin_left", "bin_right", "count", "confidence", "frequency"])
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict] = []
    total = 0.0
    n = len(labels)
    for i in range(int(n_bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == int(n_bins) - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        frequency = float(labels[mask].mean())
        total += (count / n) * abs(frequency - confidence)
        rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "count": count,
                "confidence": confidence,
                "frequency": frequency,
            }
        )
    return float(total), pd.DataFrame(rows)


def classification_reliability_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    min_rank_samples: int = MIN_IMPORTANCE_ROWS,
) -> dict:
    """
    Score a probability vector. Ranking metrics are omitted when one class is
    present or the sample is too small to be meaningful.
    """
    labels = np.asarray(y, dtype=int).ravel()
    p = np.clip(np.asarray(proba, dtype=float).ravel(), 0.0, 1.0)
    p_loss = np.clip(p, 1e-6, 1.0 - 1e-6)
    n = int(len(labels))
    empty = {
        "n": n,
        "positive_rate": float("nan"),
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "balanced_accuracy": float("nan"),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "log_loss": float("nan"),
        "ece": float("nan"),
    }
    if n == 0 or len(p) != n:
        return empty
    pred = (p >= 0.5).astype(int)
    classes = np.unique(labels)
    ece, _bins = expected_calibration_error(labels, p)
    out = {
        "n": n,
        "positive_rate": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, pred)) if len(classes) > 1 else float("nan")
        ),
        "brier": float(brier_score_loss(labels, p)),
        "log_loss": float(log_loss(labels, p_loss, labels=[0, 1])),
        "ece": float(ece),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }
    if n >= int(min_rank_samples) and len(classes) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, p))
        out["pr_auc"] = float(average_precision_score(labels, p))
    return out


def _neg_brier(estimator, X, y) -> float:
    p = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    labels = np.asarray(y, dtype=float).ravel()
    return -float(np.mean((p - labels) ** 2))


def permutation_importance_frame(
    model,
    X,
    y,
    feature_names: list[str],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance on the rows passed in.

    Those rows must be held out of fitting. Importance is the drop in negative
    Brier score when a column is shuffled. It is an association on this sample,
    not a causal effect.
    """
    names = list(feature_names)
    matrix = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int).ravel()
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.shape[0] != len(labels):
        return pd.DataFrame(columns=["Feature", "Importance", "ImportanceStd"])
    if len(labels) < MIN_IMPORTANCE_ROWS or int(n_repeats) < 1:
        return pd.DataFrame(columns=["Feature", "Importance", "ImportanceStd"])
    if not np.isfinite(matrix).all():
        return pd.DataFrame(columns=["Feature", "Importance", "ImportanceStd"])
    # Score the supplied rows only. A hand-rolled shuffle avoids sklearn's
    # requirement that the estimator implement `fit`, so a calibrator wrapper
    # can be explained the same way as the base model.
    baseline = _neg_brier(model, matrix, labels)
    rng = np.random.default_rng(int(random_state))
    means = np.empty(matrix.shape[1], dtype=float)
    stds = np.empty(matrix.shape[1], dtype=float)
    for col in range(matrix.shape[1]):
        drops = np.empty(int(n_repeats), dtype=float)
        for repeat in range(int(n_repeats)):
            shuffled = matrix.copy()
            shuffled[:, col] = rng.permutation(shuffled[:, col])
            drops[repeat] = baseline - _neg_brier(model, shuffled, labels)
        means[col] = float(drops.mean())
        stds[col] = float(drops.std(ddof=0))
    frame = pd.DataFrame({"Feature": names, "Importance": means, "ImportanceStd": stds})
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


def aggregate_importance(fold_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean importance across folds, with the fold-to-fold standard deviation."""
    tables = [t for t in fold_tables if t is not None and len(t) > 0]
    if not tables:
        return pd.DataFrame(columns=["Feature", "Importance", "FoldStd", "RepeatStd", "Folds"])
    stacked = pd.concat(tables, ignore_index=True)
    grouped = stacked.groupby("Feature", sort=False)
    out = grouped.agg(
        Importance=("Importance", "mean"),
        FoldStd=("Importance", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
        RepeatStd=("ImportanceStd", "mean"),
        Folds=("Importance", "count"),
    ).reset_index()
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def local_feature_influence(
    model,
    row_values: np.ndarray,
    reference_medians: np.ndarray,
    feature_names: list[str],
) -> tuple[float, pd.DataFrame]:
    """
    Change in probability when one feature is replaced by a held-out median.

    The delta is a sensitivity of this fitted model. It is not a causal account
    of why the market moved, and features are not independent.
    """
    names = list(feature_names)
    row = np.asarray(row_values, dtype=float).ravel()
    medians = np.asarray(reference_medians, dtype=float).ravel()
    columns = ["Feature", "HeldOutMedian", "Probability", "ReplacedProbability", "ProbabilityDelta"]
    if len(row) != len(names) or len(medians) != len(names) or not np.isfinite(row).all():
        return float("nan"), pd.DataFrame(columns=columns)
    base = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    records = []
    for i, name in enumerate(names):
        if not np.isfinite(medians[i]):
            continue
        swapped = row.copy()
        swapped[i] = medians[i]
        replaced = float(model.predict_proba(swapped.reshape(1, -1))[0, 1])
        records.append(
            {
                "Feature": name,
                "HeldOutMedian": float(medians[i]),
                "Probability": base,
                "ReplacedProbability": replaced,
                "ProbabilityDelta": base - replaced,
            }
        )
    frame = pd.DataFrame(records, columns=columns)
    if len(frame) == 0:
        return base, frame
    frame["AbsDelta"] = frame["ProbabilityDelta"].abs()
    frame = frame.sort_values(["AbsDelta", "Feature"], ascending=[False, True]).drop(columns=["AbsDelta"])
    return base, frame.reset_index(drop=True)


def _finite_slice(data: pd.DataFrame, positions: np.ndarray, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    part = data.iloc[np.asarray(positions, dtype=int)].copy()
    matrix = part[feature_cols].to_numpy(dtype=np.float64)
    mask = np.isfinite(matrix).all(axis=1)
    part = part.iloc[np.flatnonzero(mask)].reset_index()
    # `reset_index` keeps the original frame position in column "index".
    return part, matrix[mask]


def _score_method(y: np.ndarray, raw: np.ndarray, calibrator) -> tuple[np.ndarray, float]:
    if calibrator is None or len(y) == 0:
        return np.asarray([], dtype=float), float("nan")
    adjusted = apply_probability_calibrator(calibrator, raw)
    return adjusted, float(brier_score_loss(np.asarray(y, dtype=int), np.clip(adjusted, 1e-6, 1 - 1e-6)))


def evaluate_walk_forward(
    model_df: pd.DataFrame,
    cfg: ModelConfig | None = None,
    *,
    feature_cols: list[str] | None = None,
    n_folds: int = 4,
    min_train_fraction: float = 0.5,
    calibration: str = "auto",
    n_repeats: int = 5,
) -> WalkForwardEvaluation:
    """
    Fit one model per expanding window and concatenate the test-fold probabilities.

    The unlabeled tail is scored with the last successful model only. That model
    was trained before the last test fold, so its labels do not use the tail.
    """
    if cfg is None:
        cfg = ModelConfig()
    requested = _normalize_method(calibration)
    features = _resolve_feature_cols(model_df, feature_cols)
    data = model_df.sort_values("Date").reset_index(drop=True)
    labeled_pos = np.flatnonzero(data["Target"].notna().to_numpy())
    horizon = int(cfg.horizon_bars) if cfg.embargo else 0
    plans = plan_walk_forward_folds(
        len(labeled_pos),
        n_folds=n_folds,
        min_train_fraction=min_train_fraction,
        horizon_bars=horizon,
        reserve_calibration=requested != "none",
    )

    fold_rows: list[dict] = []
    oos_parts: list[pd.DataFrame] = []
    y_pools: list[np.ndarray] = []
    raw_pools: list[np.ndarray] = []
    decision_pools: list[np.ndarray] = []
    sigmoid_y: list[np.ndarray] = []
    sigmoid_p: list[np.ndarray] = []
    isotonic_y: list[np.ndarray] = []
    isotonic_p: list[np.ndarray] = []
    calib_rows: list[dict] = []
    importance_parts: list[pd.DataFrame] = []
    methods: list[str] = []

    last_base = None
    last_calibrator = None
    last_test_matrix: np.ndarray | None = None
    last_fit_end_date: pd.Timestamp | None = None
    insample_accuracy = float("nan")
    insample_n = 0

    for fold_number, plan in enumerate(plans, start=1):
        fit_pos = labeled_pos[: plan.fit_end]
        fit_frame, x_fit = _finite_slice(data, fit_pos, features)
        if len(fit_frame) < MIN_FIT_ROWS:
            continue
        y_fit = fit_frame["Target"].to_numpy(dtype=int)
        if len(np.unique(y_fit)) < 2:
            continue

        set_seed(cfg.random_state)
        base = _build_classifier(cfg)
        base.fit(x_fit, y_fit)
        fit_raw = np.asarray(base.predict_proba(x_fit)[:, 1], dtype=float)
        insample_accuracy = float(accuracy_score(y_fit, (fit_raw >= 0.5).astype(int)))
        insample_n = int(len(y_fit))

        sigmoid_cal = None
        isotonic_cal = None
        y_cal = np.asarray([], dtype=int)
        raw_cal = np.asarray([], dtype=float)
        if plan.calib_start is not None and plan.calib_end is not None:
            calib_pos = labeled_pos[plan.calib_start : plan.calib_end]
            calib_frame, x_cal = _finite_slice(data, calib_pos, features)
            if len(calib_frame) >= MIN_SIGMOID_ROWS:
                y_cal = calib_frame["Target"].to_numpy(dtype=int)
                raw_cal = np.asarray(base.predict_proba(x_cal)[:, 1], dtype=float)
                if len(np.unique(y_cal)) >= 2:
                    sigmoid_cal = fit_probability_calibrator(raw_cal, y_cal, "sigmoid")
                    isotonic_cal = fit_probability_calibrator(raw_cal, y_cal, "isotonic")

        decision_cal, decision_name = _choose_decision(requested, sigmoid_cal, isotonic_cal)

        test_pos = labeled_pos[plan.test_start : plan.test_end]
        test_frame, x_test = _finite_slice(data, test_pos, features)
        if len(test_frame) < min(8, MIN_TEST_ROWS):
            continue
        methods.append(decision_name)
        y_test = test_frame["Target"].to_numpy(dtype=int)
        raw_test = np.asarray(base.predict_proba(x_test)[:, 1], dtype=float)
        decision_test = apply_probability_calibrator(decision_cal, raw_test)
        sigmoid_test, brier_sigmoid = _score_method(y_test, raw_test, sigmoid_cal)
        isotonic_test, brier_isotonic = _score_method(y_test, raw_test, isotonic_cal)
        decision_brier = float(
            brier_score_loss(y_test, np.clip(decision_test, 1e-6, 1.0 - 1e-6))
        )
        raw_brier = float(brier_score_loss(y_test, np.clip(raw_test, 1e-6, 1.0 - 1e-6)))

        scored = test_frame.copy()
        scored["ProbaRaw"] = raw_test
        scored["ProbaCalibrated"] = decision_test if decision_name != "raw" else np.nan
        scored["Proba"] = decision_test
        scored["Fold"] = fold_number
        oos_parts.append(scored)
        y_pools.append(y_test)
        raw_pools.append(raw_test)
        decision_pools.append(decision_test)
        if len(sigmoid_test):
            sigmoid_y.append(y_test)
            sigmoid_p.append(sigmoid_test)
        if len(isotonic_test):
            isotonic_y.append(y_test)
            isotonic_p.append(isotonic_test)

        if decision_name != "raw" and len(y_cal):
            cal_decision = apply_probability_calibrator(decision_cal, raw_cal)
            calib_rows.append(
                {
                    "brier_raw": float(brier_score_loss(y_cal, np.clip(raw_cal, 1e-6, 1 - 1e-6))),
                    "brier_calibrated": float(brier_score_loss(y_cal, np.clip(cal_decision, 1e-6, 1 - 1e-6))),
                    "n": int(len(y_cal)),
                }
            )

        decision_model = _DecisionModel(base, decision_cal)
        if int(n_repeats) >= 1 and len(y_test) >= MIN_IMPORTANCE_ROWS:
            importance = permutation_importance_frame(
                decision_model,
                x_test,
                y_test,
                features,
                n_repeats=n_repeats,
                random_state=int(cfg.random_state) + fold_number,
            )
            if len(importance):
                importance = importance.copy()
                importance["Fold"] = fold_number
                importance_parts.append(importance)

        fit_last_pos = int(fit_frame["index"].iloc[-1])
        test_first_pos = int(test_frame["index"].iloc[0])
        test_last_pos = int(test_frame["index"].iloc[-1])
        calib_first_pos = None
        calib_last_pos = None
        calib_end_date = pd.NaT
        if plan.calib_start is not None and plan.calib_end is not None and len(y_cal):
            calib_first_pos = int(labeled_pos[plan.calib_start])
            calib_last_pos = int(labeled_pos[plan.calib_end - 1])
            calib_end_date = pd.Timestamp(data.loc[calib_last_pos, "Date"])

        fold_rows.append(
            {
                "fold": fold_number,
                "fit_end": int(plan.fit_end),
                "calib_start": plan.calib_start if plan.calib_start is not None else pd.NA,
                "calib_end": plan.calib_end if plan.calib_end is not None else pd.NA,
                "test_start": int(plan.test_start),
                "test_end": int(plan.test_end),
                "embargo_bars": int(plan.embargo_bars),
                "fit_last_pos": fit_last_pos,
                "calib_first_pos": calib_first_pos if calib_first_pos is not None else pd.NA,
                "calib_last_pos": calib_last_pos if calib_last_pos is not None else pd.NA,
                "test_first_pos": test_first_pos,
                "test_last_pos": test_last_pos,
                "fit_end_date": pd.Timestamp(data.loc[fit_last_pos, "Date"]),
                "calib_end_date": calib_end_date,
                "test_start_date": pd.Timestamp(test_frame["Date"].iloc[0]),
                "test_end_date": pd.Timestamp(test_frame["Date"].iloc[-1]),
                "n_fit": int(len(y_fit)),
                "n_calib": int(len(y_cal)),
                "n_test": int(len(y_test)),
                "decision_method": decision_name,
                "accuracy": float(accuracy_score(y_test, (decision_test >= 0.5).astype(int))),
                "brier_raw": raw_brier,
                "brier_decision": decision_brier,
                "brier_sigmoid": brier_sigmoid,
                "brier_isotonic": brier_isotonic,
            }
        )
        last_base = base
        last_calibrator = decision_cal
        last_test_matrix = x_test
        last_fit_end_date = pd.Timestamp(data.loc[fit_last_pos, "Date"])

    if not oos_parts or last_base is None:
        raise ValueError("Failed to create walk-forward folds. Need more history.")

    oos_frame = pd.concat(oos_parts, ignore_index=True)
    y_oos = np.concatenate(y_pools)
    raw_oos = np.concatenate(raw_pools)
    decision_oos = np.concatenate(decision_pools)
    oos_metrics = classification_reliability_metrics(y_oos, decision_oos)
    oos_raw_metrics = classification_reliability_metrics(y_oos, raw_oos)
    oos_sigmoid_metrics = (
        classification_reliability_metrics(np.concatenate(sigmoid_y), np.concatenate(sigmoid_p))
        if sigmoid_y
        else None
    )
    oos_isotonic_metrics = (
        classification_reliability_metrics(np.concatenate(isotonic_y), np.concatenate(isotonic_p))
        if isotonic_y
        else None
    )
    _ece_decision, reliability_bins = expected_calibration_error(y_oos, decision_oos)
    _ece_raw, reliability_bins_raw = expected_calibration_error(y_oos, raw_oos)
    confusion = _confusion_frame(y_oos, decision_oos)

    unique_methods = sorted(set(methods))
    if unique_methods == ["raw"]:
        decision_method = "raw"
    elif len(unique_methods) == 1:
        decision_method = unique_methods[0]
    else:
        decision_method = "mixed"

    if calib_rows:
        calib_frame = pd.DataFrame(calib_rows)
        calibration_window_metrics = {
            "brier_raw": float(calib_frame["brier_raw"].mean()),
            "brier_calibrated": float(calib_frame["brier_calibrated"].mean()),
            "n_mean": float(calib_frame["n"].mean()),
            "folds": int(len(calib_frame)),
            "note": "Calibration-window scores reuse the rows the calibrator was fit on. They are not out-of-sample.",
        }
    else:
        calibration_window_metrics = None

    last_oos_pos = int(oos_frame["index"].max()) if "index" in oos_frame.columns else int(labeled_pos[-1])
    tail = data.iloc[last_oos_pos + 1 :].copy()
    tail_matrix = tail[features].to_numpy(dtype=np.float64) if len(tail) else np.empty((0, len(features)))
    if len(tail):
        tail_mask = np.isfinite(tail_matrix).all(axis=1)
        tail = tail.iloc[np.flatnonzero(tail_mask)].copy()
        tail_matrix = tail_matrix[tail_mask]
    if len(tail):
        tail_raw = np.asarray(last_base.predict_proba(tail_matrix)[:, 1], dtype=float)
        tail_decision = apply_probability_calibrator(last_calibrator, tail_raw)
        tail = tail.reset_index()
        tail["ProbaRaw"] = tail_raw
        tail["ProbaCalibrated"] = tail_decision if last_calibrator is not None else np.nan
        tail["Proba"] = tail_decision
        tail["Fold"] = 0
    else:
        tail = pd.DataFrame()

    explained_date = None
    explained_probability = float("nan")
    explained_position = -1
    local = pd.DataFrame()
    if last_test_matrix is not None and len(last_test_matrix) >= 5:
        medians = np.nanmedian(last_test_matrix, axis=0)
        if len(tail):
            explain_row = tail.iloc[-1]
            row_values = explain_row[features].to_numpy(dtype=float)
        else:
            explain_row = oos_frame.iloc[-1]
            row_values = np.asarray(last_test_matrix[-1], dtype=float)
        decision_model = _DecisionModel(last_base, last_calibrator)
        explained_probability, local = local_feature_influence(decision_model, row_values, medians, features)
        explained_date = pd.Timestamp(explain_row["Date"])
        explained_position = int(explain_row["index"])

    folds = pd.DataFrame(fold_rows)
    return WalkForwardEvaluation(
        folds=folds,
        oos_frame=oos_frame,
        tail_frame=tail,
        oos_y=y_oos,
        oos_raw=raw_oos,
        oos_decision=decision_oos,
        oos_metrics=oos_metrics,
        oos_raw_metrics=oos_raw_metrics,
        oos_sigmoid_metrics=oos_sigmoid_metrics,
        oos_isotonic_metrics=oos_isotonic_metrics,
        insample_accuracy=insample_accuracy,
        insample_n=insample_n,
        calibration_window_metrics=calibration_window_metrics,
        reliability_bins=reliability_bins,
        reliability_bins_raw=reliability_bins_raw,
        confusion=confusion,
        importance=aggregate_importance(importance_parts),
        local_explanation=local,
        calibration_requested=requested,
        decision_method=decision_method,
        explained_date=explained_date,
        explained_probability=explained_probability,
        explained_position=explained_position,
        last_fit_end_date=last_fit_end_date,
    )


def _confusion_frame(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(y, dtype=int).ravel()
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return pd.DataFrame()
    pred = (np.asarray(proba, dtype=float).ravel() >= 0.5).astype(int)
    matrix = confusion_matrix(labels, pred, labels=[0, 1])
    return pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])


def assemble_decision_frame(result: WalkForwardEvaluation) -> pd.DataFrame:
    """Out-of-sample rows followed by the unlabeled tail, in time order."""
    parts = [result.oos_frame]
    if result.tail_frame is not None and len(result.tail_frame):
        parts.append(result.tail_frame)
    frame = pd.concat(parts, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def public_reliability_dict(result: WalkForwardEvaluation) -> dict:
    """JSON-friendly view for the research UI. Estimators are not included."""
    return {
        "evaluation": "walk_forward",
        "calibration_requested": result.calibration_requested,
        "decision_method": result.decision_method,
        "folds": result.folds,
        "oos_metrics": result.oos_metrics,
        "oos_raw_metrics": result.oos_raw_metrics,
        "oos_sigmoid_metrics": result.oos_sigmoid_metrics,
        "oos_isotonic_metrics": result.oos_isotonic_metrics,
        "insample_accuracy": result.insample_accuracy,
        "insample_n": result.insample_n,
        "calibration_window_metrics": result.calibration_window_metrics,
        "reliability_bins": result.reliability_bins,
        "reliability_bins_raw": result.reliability_bins_raw,
        "confusion": result.confusion,
        "importance": result.importance,
        "local_explanation": result.local_explanation,
        "oos_proba_raw": result.oos_raw,
        "oos_proba_decision": result.oos_decision,
        "explained_date": result.explained_date,
        "explained_probability": result.explained_probability,
        "last_fit_end_date": result.last_fit_end_date,
        "n_oos": int(result.oos_metrics.get("n", 0)),
    }


def holdout_reliability_dict(reason: str) -> dict:
    return {
        "evaluation": "holdout",
        "reason": reason,
        "decision_method": "raw",
        "calibration_requested": "none",
    }
