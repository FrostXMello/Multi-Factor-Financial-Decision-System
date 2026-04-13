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


@dataclass
class ModelResult:
    model: object
    test_df: pd.DataFrame
    test_proba: np.ndarray  # P(Target=1) for each row in test_df
    test_pred: np.ndarray   # predicted class labels
    test_accuracy: float


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
            n_estimators=300,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_type in ("gb", "hgb", "hist_gradient_boosting"):
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=250,
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


def train_model(
    model_df: pd.DataFrame,
    cfg: ModelConfig | None = None,
    *,
    feature_cols: list[str] | None = None,
) -> ModelResult:
    """
    Train a classifier on historical features and return next-day probabilities
    for the out-of-sample (test) period.

    Uses a chronological split (first 80% train, last 20% test) to avoid leakage.
    """
    if cfg is None:
        cfg = ModelConfig()

    if cfg.test_fraction <= 0 or cfg.test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1")

    train_features = _resolve_feature_cols(model_df, feature_cols)

    data = model_df.sort_values("Date").reset_index(drop=True)
    n = len(data)
    min_rows = int(cfg.min_rows)
    if n < min_rows:
        raise ValueError(f"Not enough data to train (rows={n}, min_rows={min_rows}). Try another ticker.")

    split_idx = int((1 - cfg.test_fraction) * n)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    X_train = train_df[train_features].values
    y_train = train_df["Target"].values
    X_test = test_df[train_features].values
    y_test = test_df["Target"].values

    set_seed(cfg.random_state)

    clf = _build_classifier(cfg)

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, pred))

    return ModelResult(
        model=clf,
        test_df=test_df,
        test_proba=proba,
        test_pred=pred,
        test_accuracy=acc,
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
    n = len(data)
    min_train_size = int(n * min_train_fraction)
    if min_train_size < 100:
        raise ValueError("Not enough rows for walk-forward validation")

    remaining = n - min_train_size
    step = max(1, remaining // n_folds)
    scores: list[float] = []

    start_test = min_train_size
    while start_test < n and len(scores) < n_folds:
        end_test = min(n, start_test + step)
        train_df = data.iloc[:start_test]
        test_df = data.iloc[start_test:end_test]
        if len(test_df) < 20:
            break

        clf = _build_classifier(cfg)
        X_train = train_df[train_features].values
        y_train = train_df["Target"].values
        X_test = test_df[train_features].values
        y_test = test_df["Target"].values
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

