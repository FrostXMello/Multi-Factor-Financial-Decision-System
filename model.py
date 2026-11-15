from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import set_seed


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "gb"  # "logreg", "rf", "gb"
    test_fraction: float = 0.2
    random_state: int = 42
    # Minimum rows required to train (multi-layer scans may use shorter histories per name).
    min_rows: int = 200
    # Label horizon. Training rows whose forward window reaches the test period are dropped.
    horizon_bars: int = 1
    embargo: bool = True
    # Estimator size. Defaults match the previous fixed HistGradientBoosting / forest settings.
    max_iter: int = 250
    n_estimators: int = 300


@dataclass
class ModelResult:
    model: object
    test_df: pd.DataFrame
    test_proba: np.ndarray  # P(Target=1) for each row in test_df
    test_pred: np.ndarray   # predicted class labels
    test_accuracy: float
    train_end_date: pd.Timestamp | None = None
    test_start_date: pd.Timestamp | None = None
    horizon_bars: int = 1
    embargo_bars: int = 0


def purged_split_bounds(n_labeled: int, test_fraction: float, horizon_bars: int) -> tuple[int, int]:
    """
    Chronological split on labeled rows.

    Returns `(train_end_exclusive, test_start)` in labeled-row coordinates.
    Rows in `[train_end_exclusive, test_start)` are the embargo: their labels
    use closes that fall inside the test window, so they are neither trained
    on nor scored as the start of the test.
    """
    if n_labeled < 2:
        raise ValueError(f"Need at least 2 labeled rows to split (got {n_labeled})")
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1")
    horizon = max(0, int(horizon_bars))
    split = int((1.0 - float(test_fraction)) * n_labeled)
    split = min(max(split, 1), n_labeled - 1)
    train_end = split - horizon
    if train_end < 1:
        raise ValueError(
            f"Embargo of {horizon} bars leaves no training rows "
            f"(labeled={n_labeled}, test starts at {split})."
        )
    return train_end, split


# Default legacy CORE / multifactor base columns; TECHNICAL_MODE uses the list returned from
# `build_multi_timeframe_dataset` (includes Price_vs_MA50, ROC_*, optional TF_* weekly).
FEATURE_COLS = ["MA20", "MA50", "RSI14", "Daily_Return", "Rolling_Volatility_10"]


@dataclass
class WalkForwardResult:
    fold_scores: list[float]
    mean_score: float
    folds: int


def _build_classifier(cfg: ModelConfig):
    model_type = cfg.model_type.lower().strip()
    if model_type == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=cfg.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    if model_type in ("rf", "randomforest", "random_forest"):
        return RandomForestClassifier(
            n_estimators=int(cfg.n_estimators),
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_type in ("gb", "hgb", "hist_gradient_boosting"):
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=int(cfg.max_iter),
            random_state=cfg.random_state,
        )
    raise ValueError("model_type must be 'logreg', 'rf', or 'gb'")


def _resolve_feature_cols(model_df: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    cols = FEATURE_COLS if feature_cols is None else list(feature_cols)
    if not cols:
        raise ValueError("feature_cols cannot be empty")
    missing = [c for c in cols + ["Target"] if c not in model_df.columns]
    if missing:
        raise ValueError(f"model_df missing columns: {missing}")
    return cols


def score_features(model: object, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Score rows from feature columns only.

    `Target` is not read. The latest bar can be scored before its forward return exists.
    """
    cols = list(feature_cols)
    if not cols:
        raise ValueError("feature_cols cannot be empty")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns: {missing}")
    X = frame[cols].replace([np.inf, -np.inf], np.nan)
    if X.isna().any(axis=None):
        raise ValueError("score_features requires finite feature values on every row")
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba")
    proba = model.predict_proba(X.values)[:, 1]
    return np.asarray(proba, dtype=float)


def train_model(
    model_df: pd.DataFrame,
    cfg: ModelConfig | None = None,
    *,
    feature_cols: list[str] | None = None,
) -> ModelResult:
    """
    Train on completed labels only and score the held-out suffix, including the unlabeled tail.

    Chronological split plus an embargo of `horizon_bars` so a training label cannot
    use a close from the test period. Accuracy is computed only on rows with a Target.
    """
    if cfg is None:
        cfg = ModelConfig()

    train_features = _resolve_feature_cols(model_df, feature_cols)

    data = model_df.sort_values("Date").reset_index(drop=True)
    labeled_pos = np.flatnonzero(data["Target"].notna().to_numpy())
    n_labeled = int(len(labeled_pos))
    min_rows = int(cfg.min_rows)
    if n_labeled < min_rows:
        raise ValueError(
            f"Not enough completed labels to train (labeled={n_labeled}, min_rows={min_rows}). "
            "Try another ticker."
        )

    horizon = int(cfg.horizon_bars) if cfg.embargo else 0
    train_end, test_start = purged_split_bounds(n_labeled, cfg.test_fraction, horizon)
    train_pos = labeled_pos[:train_end]
    test_pos0 = int(labeled_pos[test_start])
    train_df = data.iloc[train_pos].copy()
    # Suffix from the first test label through the live tail (Target may be null).
    test_df = data.iloc[test_pos0:].copy().reset_index(drop=True)

    if len(train_df) < 30:
        raise ValueError(
            f"Embargo left only {len(train_df)} training rows. Reduce horizon_bars or add history."
        )

    y_train = train_df["Target"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels contain a single class after the embargo split.")

    set_seed(cfg.random_state)
    clf = _build_classifier(cfg)
    clf.fit(train_df[train_features].values, y_train)

    proba = score_features(clf, test_df, train_features)
    pred = (proba >= 0.5).astype(int)
    labeled_mask = test_df["Target"].notna().to_numpy()
    if int(labeled_mask.sum()) == 0:
        acc = float("nan")
    else:
        y_test = test_df.loc[labeled_mask, "Target"].astype(int).to_numpy()
        acc = float(accuracy_score(y_test, pred[labeled_mask]))

    return ModelResult(
        model=clf,
        test_df=test_df,
        test_proba=proba,
        test_pred=pred,
        test_accuracy=acc,
        train_end_date=pd.Timestamp(train_df["Date"].iloc[-1]),
        test_start_date=pd.Timestamp(test_df["Date"].iloc[0]),
        horizon_bars=int(cfg.horizon_bars),
        embargo_bars=horizon,
    )


def walk_forward_validate(
    model_df: pd.DataFrame,
    cfg: ModelConfig | None = None,
    *,
    feature_cols: list[str] | None = None,
    n_folds: int = 5,
    min_train_fraction: float = 0.5,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward validation for time-series robustness.
    """
    if cfg is None:
        cfg = ModelConfig()
    if n_folds <= 0:
        raise ValueError("n_folds must be >= 1")
    if not (0.2 <= min_train_fraction < 0.95):
        raise ValueError("min_train_fraction must be in [0.2, 0.95)")

    train_features = _resolve_feature_cols(model_df, feature_cols)

    data = model_df.sort_values("Date").reset_index(drop=True)
    labeled_pos = np.flatnonzero(data["Target"].notna().to_numpy())
    labeled = data.iloc[labeled_pos].reset_index(drop=True)
    n = len(labeled)
    horizon = int(cfg.horizon_bars) if cfg.embargo else 0
    min_train_size = int(n * min_train_fraction)
    if min_train_size < 100:
        raise ValueError("Not enough rows for walk-forward validation")

    remaining = n - min_train_size
    step = max(1, remaining // n_folds)
    scores: list[float] = []

    start_test = min_train_size
    while start_test < n and len(scores) < n_folds:
        end_test = min(n, start_test + step)
        train_end = start_test - horizon
        if train_end < 30:
            break
        train_df = labeled.iloc[:train_end]
        test_df = labeled.iloc[start_test:end_test]
        if len(test_df) < 20:
            break
        y_train = train_df["Target"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            start_test = end_test
            continue

        clf = _build_classifier(cfg)
        X_train = train_df[train_features].values
        X_test = test_df[train_features].values
        y_test = test_df["Target"].astype(int).to_numpy()
        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
        else:
            pred = clf.predict(X_test)
        score = float(accuracy_score(y_test, pred))
        scores.append(score)
        start_test = end_test

    if not scores:
        raise ValueError("Failed to create walk-forward folds. Need more history.")

    return WalkForwardResult(
        fold_scores=scores,
        mean_score=float(np.mean(scores)),
        folds=len(scores),
    )


def feature_importance_dataframe(model: object, feature_names: list[str]) -> pd.DataFrame:
    """
    Sorted importance / coefficient magnitude for tree models or logistic regression (inside a Pipeline or raw).
    """
    try:
        est = model
        if isinstance(model, Pipeline):
            if "lr" in model.named_steps:
                est = model.named_steps["lr"]
            else:
                est = model.named_steps[list(model.named_steps.keys())[-1]]
        if hasattr(est, "feature_importances_"):
            v = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            v = np.abs(np.ravel(est.coef_))
        else:
            return pd.DataFrame()
        if len(v) != len(feature_names):
            return pd.DataFrame()
        df = pd.DataFrame({"Feature": list(feature_names), "Importance": v})
        df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
        s = float(df["Importance"].sum())
        df["ImportancePct"] = (df["Importance"] / s * 100.0) if s > 0 else 0.0
        return df
    except Exception:
        return pd.DataFrame()

