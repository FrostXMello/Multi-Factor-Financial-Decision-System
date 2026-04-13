"""
Three-layer quant-style stock selection.

Layer 1 — Fast filtering: trend, liquidity, volatility bounds, momentum ranking.
Layer 2 — ML scoring: same technical feature stack as TECHNICAL_MODE; chronological GB + proba.
Layer 3 — Portfolio: probability & MA50 gates, sector diversification, optional correlation cap.

Comments in each function document the economic intent for academic demos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from model import ModelConfig, train_model
from multi_layer_config import (
    LAYER1_MA_TREND,
    LAYER1_MAX_ANN_VOL,
    LAYER1_MIN_ANN_VOL,
    LAYER1_MOMENTUM_DAYS,
    LAYER1_VOL_WINDOW,
    MultiLayerPipelineConfig,
)
from multi_layer_data import _avg_dollar_volume_20d


def infer_sector(ticker: str) -> str:
    """
    Coarse sector bucket for diversification (no live sector API — keeps scans fast).
    Extend `SECTOR_KEYWORDS` for finer splits if needed.
    """
    t = ticker.upper().replace(".NS", "").replace(".BO", "")
    rules: list[tuple[tuple[str, ...], str]] = (
        (("BANK", "FIN", "HDFC", "BAJAJ", "CHOLA", "MUTHOOT", "MANAPPURAM", "SHRIRAM"), "Financials"),
        (("TECH", "INFY", "WIPRO", "HCL", "MPHASIS", "COFORGE", "PERSISTENT", "LTIM", "LTTS"), "IT"),
        (("OIL", "ONGC", "GAIL", "PETRO", "IOC", "BPCL", "OIL"), "Energy"),
        (("PHARMA", "DR", "CIPLA", "SUN", "LUPIN", "BIO", "AURO", "ALKEM", "ZYDUS"), "Pharma"),
        (("AUTO", "MARUTI", "MOTOR", "EICHER", "BAJAJ-A", "HERO", "TVS", "ASHOK", "M&M"), "Autos"),
        (("STEEL", "JSW", "TATAST", "SAIL", "JINDAL", "VEDL", "HINDAL"), "Metals"),
        (("CEMENT", "ULTRA", "SHREE", "ACC", "AMBUJA", "RAMCO"), "Cement"),
        (("POWER", "NTPC", "POWERGRID", "TATAPOWER", "ADANI", "TORNTPOWER"), "Utilities_Power"),
        (("FMCG", "ITC", "HINDUN", "NESTLE", "BRIT", "DABUR", "MARICO", "TATACON"), "FMCG"),
        (("REAL", "DLF", "LODHA", "OBEROI", "PRESTIG", "SOBHA"), "RealEstate"),
        (("TELECOM", "BHARTI", "IDEA"), "Telecom"),
    )
    for keys, name in rules:
        if any(k in t for k in keys):
            return name
    return "Other"


def layer1_filter(
    prices_by_ticker: dict[str, pd.DataFrame],
    *,
    out_max: int = 60,
    out_min: int = 40,
    momentum_days: int = LAYER1_MOMENTUM_DAYS,
    ma_window: int = LAYER1_MA_TREND,
    vol_window: int = LAYER1_VOL_WINDOW,
    min_ann_vol: float = LAYER1_MIN_ANN_VOL,
    max_ann_vol: float = LAYER1_MAX_ANN_VOL,
    min_dollar_vol_20d: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Layer 1 — Broad, cheap filters to drop illiquid / chaotic names and weak trends.

    Keeps names with:
      - Close > MA50 (long-only academic demo)
      - 20d annualized vol inside [min_ann_vol, max_ann_vol]
      - Recent dollar liquidity above gate (uses last 20d avg Close*Volume)

    Ranks survivors by 20d total return (momentum) and returns the top `out_max` names
    (but at least tries to keep `out_min` if that many pass — if fewer pass, returns all).
    """
    from multi_layer_config import MIN_AVG_DOLLAR_VOLUME_20D

    min_dv = float(min_dollar_vol_20d) if min_dollar_vol_20d is not None else float(MIN_AVG_DOLLAR_VOLUME_20D)
    rows: list[dict] = []
    need = max(ma_window, momentum_days, vol_window) + 5

    for sym, df in prices_by_ticker.items():
        if df is None or len(df) < need:
            continue
        d = df.copy()
        c = d["Close"].astype(float)
        ma = c.rolling(ma_window, min_periods=ma_window).mean()
        ret1 = c.pct_change()
        ann_vol = ret1.rolling(vol_window, min_periods=vol_window).std().iloc[-1] * float(np.sqrt(252))
        mom = c.pct_change(momentum_days).iloc[-1]
        last_px = float(c.iloc[-1])
        last_ma = float(ma.iloc[-1])
        if not np.isfinite(last_ma) or last_ma <= 0 or not np.isfinite(last_px):
            continue
        if last_px <= last_ma:
            continue
        if not np.isfinite(ann_vol) or ann_vol < min_ann_vol or ann_vol > max_ann_vol:
            continue
        dv = _avg_dollar_volume_20d(d)
        if dv < min_dv:
            continue
        if not np.isfinite(mom):
            continue
        rows.append(
            {
                "Ticker": sym,
                "Momentum20d": float(mom),
                "AnnVol": float(ann_vol),
                "Close": last_px,
                "MA50": last_ma,
                "DollarVol20d": dv,
            }
        )

    if not rows:
        return [], pd.DataFrame()

    tab = pd.DataFrame(rows).sort_values("Momentum20d", ascending=False).reset_index(drop=True)
    # Take top out_max; if we have fewer than out_min pass, return all that passed.
    k = min(out_max, len(tab))
    if k < out_min and len(tab) >= out_min:
        k = out_min
    tab = tab.head(k).reset_index(drop=True)
    return tab["Ticker"].tolist(), tab


def layer2_model_scores(
    tickers: list[str],
    prices_by_ticker: dict[str, pd.DataFrame],
    cfg: MultiLayerPipelineConfig | None = None,
) -> pd.DataFrame:
    """
    Layer 2 — Per-name gradient boosting on the technical-only label (forward > ~1% over horizon).

    Uses the **last chronological test-fold probability** as the live score (no peeking at future
    rows beyond the train/test split inside each series). Optional risk-adjusted score divides by
    recent annualized vol to penalize fragile high-beta names.
    """
    if cfg is None:
        cfg = MultiLayerPipelineConfig()
    out_rows: list[dict] = []

    for sym in tickers:
        df = prices_by_ticker.get(sym)
        if df is None or len(df) < cfg.model_min_rows + cfg.horizon_bars + 5:
            continue
        try:
            tf = {"1d": df}
            model_df, feat_cols = build_multi_timeframe_dataset(
                tf,
                cfg=MultiTimeframeDatasetConfig(
                    base_interval="1d",
                    context_intervals=(),
                    horizon_bars=cfg.horizon_bars,
                    technical_only=True,
                    technical_min_forward_return=cfg.min_forward_return,
                ),
            )
            res = train_model(
                model_df,
                cfg=ModelConfig(model_type="gb", min_rows=cfg.model_min_rows, random_state=42),
                feature_cols=feat_cols,
            )
            proba = float(res.test_proba[-1])
            last_test = res.test_df.iloc[-1]
            rv = float(last_test.get("Rolling_Volatility_10", np.nan))
            ann_vol = float(rv * np.sqrt(252)) if np.isfinite(rv) and rv > 0 else float("nan")
            score = proba / max(ann_vol, 1e-6) if cfg.use_risk_adjusted and np.isfinite(ann_vol) else proba
            out_rows.append(
                {
                    "Ticker": sym,
                    "Proba": proba,
                    "AnnVol": ann_vol,
                    "Score": float(score),
                    "TestAccuracy": float(res.test_accuracy),
                    "MA50": float(last_test["MA50"]) if "MA50" in last_test.index else float("nan"),
                    "Close": float(last_test["Close"]) if "Close" in last_test.index else float("nan"),
                }
            )
        except Exception:
            continue

    if not out_rows:
        return pd.DataFrame()
    scored = pd.DataFrame(out_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return scored.head(int(cfg.layer2_top_k)).reset_index(drop=True)


def _pairwise_max_corr(
    prices_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    lookback: int = 60,
) -> pd.DataFrame:
    rets: dict[str, pd.Series] = {}
    for t in tickers:
        df = prices_by_ticker.get(t)
        if df is None or len(df) < lookback + 2:
            continue
        c = df["Close"].astype(float).tail(lookback + 1)
        rets[t] = np.log(c).diff().dropna()
    if len(rets) < 2:
        return pd.DataFrame()
    aligned = pd.DataFrame(rets).dropna(how="any", axis=0)
    if aligned.shape[1] < 2 or len(aligned) < 10:
        return pd.DataFrame()
    return aligned.corr()


def layer3_select_portfolio(
    scored: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    cfg: MultiLayerPipelineConfig | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Layer 3 — Turn ranked scores into a tradable basket.

    Rules:
      - Probability must exceed `min_proba` and price must confirm vs MA50 on the latest row.
      - Greedy fill up to `layer3_n` names while enforcing **one name per coarse sector**.
      - Optional: drop a candidate if max correlation vs already picked exceeds `max_pairwise_corr`.
    """
    if cfg is None:
        cfg = MultiLayerPipelineConfig()
    if scored is None or scored.empty:
        return [], scored

    candidates = scored[scored["Proba"] >= float(cfg.min_proba)].copy()
    # Refresh price > MA50 from latest raw prices (handles slight misalignment vs test row).
    ok: list[str] = []
    for _, row in candidates.iterrows():
        sym = str(row["Ticker"])
        df = prices_by_ticker.get(sym)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        px = float(last["Close"])
        c = df["Close"].astype(float)
        ma50 = float(c.rolling(50, min_periods=50).mean().iloc[-1])
        if np.isfinite(ma50) and px > ma50:
            ok.append(sym)
    candidates = candidates[candidates["Ticker"].isin(ok)].reset_index(drop=True)
    if candidates.empty:
        return [], scored

    picked: list[str] = []
    sectors_used: set[str] = set()
    corr_mat = (
        _pairwise_max_corr(prices_by_ticker, candidates["Ticker"].tolist())
        if cfg.max_pairwise_corr is not None
        else pd.DataFrame()
    )

    for _, row in candidates.iterrows():
        sym = str(row["Ticker"])
        if len(picked) >= int(cfg.layer3_n):
            break
        sec = infer_sector(sym)
        if sec in sectors_used:
            continue
        if cfg.max_pairwise_corr is not None and picked and not corr_mat.empty and sym in corr_mat.index:
            too_correlated = False
            for existing in picked:
                if existing in corr_mat.columns:
                    v = abs(float(corr_mat.loc[sym, existing]))
                    if np.isfinite(v) and v > float(cfg.max_pairwise_corr):
                        too_correlated = True
                        break
            if too_correlated:
                continue
        picked.append(sym)
        sectors_used.add(sec)

    detail = candidates[candidates["Ticker"].isin(picked)].copy()
    return picked, detail


# Public aliases (requested API shape)
layer2_model = layer2_model_scores
layer3_selection = layer3_select_portfolio


def run_multi_layer_pipeline(
    prices_by_ticker: dict[str, pd.DataFrame],
    cfg: MultiLayerPipelineConfig | None = None,
) -> dict:
    """
    Run Layer 1 → 2 → 3 once on the latest available data for each series.
    """
    if cfg is None:
        cfg = MultiLayerPipelineConfig()
    l1_tickers, l1_tab = layer1_filter(
        prices_by_ticker,
        out_max=int(cfg.layer1_out_max),
        out_min=int(cfg.layer1_out_min),
    )
    l2_tab = layer2_model_scores(l1_tickers, prices_by_ticker, cfg=cfg)
    final_tickers, l3_tab = layer3_select_portfolio(l2_tab, prices_by_ticker, cfg=cfg)
    return {
        "layer1_table": l1_tab,
        "layer1_tickers": l1_tickers,
        "layer2_table": l2_tab,
        "layer3_table": l3_tab,
        "final_tickers": final_tickers,
    }
