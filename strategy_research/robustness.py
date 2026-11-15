"""Walk-forward robustness summaries. Diagnostic only — no test-set optimization."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def walk_forward_robustness(
    fold_scores: Optional[list] = None,
    *,
    folds_frame: Optional[pd.DataFrame] = None,
    score_column: str = "accuracy",
) -> dict[str, Any]:
    """
    Summarize purged walk-forward fold scores already produced by model evaluation.

    Does not retrain models and does not tune thresholds on the held-out folds.
    """
    scores: list[float] = []
    if folds_frame is not None and isinstance(folds_frame, pd.DataFrame) and score_column in folds_frame.columns:
        scores = [float(x) for x in folds_frame[score_column].tolist() if np.isfinite(float(x))]
    elif fold_scores is not None:
        for value in fold_scores:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                scores.append(number)

    if len(scores) < 2:
        return {
            "available": False,
            "detail": "insufficient_data",
            "n_folds": len(scores),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "consistency_ratio": None,
            "weaknesses": ["Fewer than two finite walk-forward folds are available."],
            "note": "Robustness is diagnostic; fold scores are not used here to retune the strategy.",
        }

    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    consistency = float(mean / std) if std > 1e-12 else None
    weaknesses = []
    if std > 0.15:
        weaknesses.append("Fold scores are highly dispersed; performance may be regime-sensitive.")
    if float(arr.min()) < 0.45:
        weaknesses.append("At least one fold accuracy is below 0.45.")
    if consistency is not None and consistency < 2.0:
        weaknesses.append("Mean/std consistency ratio is low across folds.")

    return {
        "available": True,
        "detail": "ok",
        "n_folds": int(len(arr)),
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "consistency_ratio": consistency,
        "fold_scores": scores,
        "weaknesses": weaknesses,
        "note": (
            "Walk-forward robustness uses existing out-of-sample fold metrics only. "
            "No parameter search is performed against these folds."
        ),
    }
