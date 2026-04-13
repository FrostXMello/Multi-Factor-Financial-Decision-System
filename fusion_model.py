from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class FusionConfig:
    macro_weight: float = 0.25
    micro_weight: float = 0.40
    tech_weight: float = 0.35
    random_state: int = 42


def _check_weights(cfg: FusionConfig) -> None:
    s = cfg.macro_weight + cfg.micro_weight + cfg.tech_weight
    if not (0.999 <= s <= 1.001):
        raise ValueError("Fusion weights must sum to 1.0")


def core_simple_fusion(
    tech_proba: pd.Series,
    macro_proba: pd.Series,
    *,
    include_macro: bool,
    tech_weight: float = 0.7,
    macro_weight: float = 0.3,
) -> pd.Series:
    """
    Non-trainable fusion for CORE_MODE: either technical only or fixed Tech/Macro weights.
    """
    tech = tech_proba.astype(float).clip(0.0, 1.0)
    if not include_macro:
        return tech
    w_t = float(tech_weight)
    w_m = float(macro_weight)
    if abs(w_t + w_m - 1.0) > 1e-3:
        raise ValueError("tech_weight + macro_weight must sum to 1.0")
    macro = macro_proba.astype(float).clip(0.0, 1.0)
    return (w_t * tech + w_m * macro).clip(0.0, 1.0)


def weighted_fusion(
    macro_proba: pd.Series,
    micro_proba: pd.Series,
    tech_proba: pd.Series,
    cfg: FusionConfig | None = None,
) -> pd.Series:
    if cfg is None:
        cfg = FusionConfig()
    _check_weights(cfg)
    fused = cfg.macro_weight * macro_proba + cfg.micro_weight * micro_proba + cfg.tech_weight * tech_proba
    return fused.clip(lower=0.0, upper=1.0)


def meta_fusion(
    features_df: pd.DataFrame,
    target: pd.Series,
    cfg: FusionConfig | None = None,
) -> pd.Series:
    """
    Train a simple meta-model on stacked probabilities and return in-sample probabilities.
    """
    if cfg is None:
        cfg = FusionConfig()
    needed = ["MacroProba", "MicroProba", "TechProba"]
    missing = [c for c in needed if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df missing columns: {missing}")

    X = features_df[needed].values
    y = target.values
    clf = LogisticRegression(max_iter=2000, random_state=cfg.random_state, class_weight="balanced")
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    return pd.Series(proba, index=features_df.index, name="AlphaProba")
