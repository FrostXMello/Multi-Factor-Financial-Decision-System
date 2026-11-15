"""
Configurable cash-equity transaction costs for Indian backtests.

These defaults are model assumptions for NSE cash equity. They are not a live
broker tariff and they are not fetched from an exchange circular. Override any
rate before treating a result as a cost estimate for a specific broker.

Statutory-style structure used by the presets (researcher should confirm before
relying on the numbers):

- Brokerage: a percentage of traded notional. The delivery preset uses 3 bps
  per side as a conservative research default. Many discount brokers charge 0
  on delivery; set `brokerage_rate=0` for that case.
- STT: delivery preset charges 0.10% on buy and 0.10% on sell. Intraday preset
  charges 0.025% on the sell side only.
- Exchange transaction charge: 0.00297% per side. NSE revises this; treat it as
  an assumption.
- SEBI charge: 0.0001% per side (₹10 per crore of notional).
- Stamp duty: delivery buy 0.015%, intraday buy 0.003%, sell 0 in both presets.
- GST: 18% of (brokerage + exchange transaction charge + SEBI charge). GST is
  not applied to STT or stamp duty.
- Slippage: basis points added to the buy fill and subtracted from the sell fill.
  Slippage is a price adjustment, not part of the statutory charge total.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class IndiaEquityCostConfig:
    """Rates are fractions of notional unless noted. `slippage_bps` is basis points."""

    segment: str = "equity_delivery"
    brokerage_rate: float = 0.0003
    stt_buy_rate: float = 0.001
    stt_sell_rate: float = 0.001
    exchange_rate: float = 0.0000297
    sebi_rate: float = 0.000001
    stamp_buy_rate: float = 0.00015
    stamp_sell_rate: float = 0.0
    gst_rate: float = 0.18
    slippage_bps: float = 5.0

    @staticmethod
    def equity_delivery(**overrides: float) -> "IndiaEquityCostConfig":
        cfg = IndiaEquityCostConfig(segment="equity_delivery")
        return replace(cfg, **overrides) if overrides else cfg

    @staticmethod
    def equity_intraday(**overrides: float) -> "IndiaEquityCostConfig":
        cfg = IndiaEquityCostConfig(
            segment="equity_intraday",
            brokerage_rate=0.0003,
            stt_buy_rate=0.0,
            stt_sell_rate=0.00025,
            stamp_buy_rate=0.00003,
            stamp_sell_rate=0.0,
            slippage_bps=5.0,
        )
        return replace(cfg, **overrides) if overrides else cfg

    @staticmethod
    def zero() -> "IndiaEquityCostConfig":
        return IndiaEquityCostConfig(
            segment="zero",
            brokerage_rate=0.0,
            stt_buy_rate=0.0,
            stt_sell_rate=0.0,
            exchange_rate=0.0,
            sebi_rate=0.0,
            stamp_buy_rate=0.0,
            stamp_sell_rate=0.0,
            gst_rate=0.0,
            slippage_bps=0.0,
        )


def cost_config_for_segment(segment: str, **overrides: float) -> IndiaEquityCostConfig:
    key = str(segment).strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"equity_delivery", "delivery", "cnc"}:
        return IndiaEquityCostConfig.equity_delivery(**overrides)
    if key in {"equity_intraday", "intraday", "mis"}:
        return IndiaEquityCostConfig.equity_intraday(**overrides)
    if key in {"zero", "none"}:
        return IndiaEquityCostConfig.zero()
    raise ValueError("segment must be 'equity_delivery', 'equity_intraday', or 'zero'")


def _gst_base(notional: float, cfg: IndiaEquityCostConfig) -> float:
    return float(notional) * (cfg.brokerage_rate + cfg.exchange_rate + cfg.sebi_rate)


def buy_charges(notional: float, cfg: IndiaEquityCostConfig | None = None) -> float:
    """Cash charges on a buy, excluding slippage (slippage is in the fill price)."""
    if cfg is None:
        cfg = IndiaEquityCostConfig.equity_delivery()
    gross = max(0.0, float(notional))
    statutory = gross * (cfg.brokerage_rate + cfg.stt_buy_rate + cfg.exchange_rate + cfg.sebi_rate + cfg.stamp_buy_rate)
    gst = cfg.gst_rate * _gst_base(gross, cfg)
    return float(statutory + gst)


def sell_charges(notional: float, cfg: IndiaEquityCostConfig | None = None) -> float:
    """Cash charges on a sell, excluding slippage."""
    if cfg is None:
        cfg = IndiaEquityCostConfig.equity_delivery()
    gross = max(0.0, float(notional))
    statutory = gross * (
        cfg.brokerage_rate + cfg.stt_sell_rate + cfg.exchange_rate + cfg.sebi_rate + cfg.stamp_sell_rate
    )
    gst = cfg.gst_rate * _gst_base(gross, cfg)
    return float(statutory + gst)


def buy_cost_fraction(cfg: IndiaEquityCostConfig | None = None) -> float:
    """Charges per unit of buy notional. Linear in the presets above."""
    return buy_charges(1.0, cfg)


def sell_cost_fraction(cfg: IndiaEquityCostConfig | None = None) -> float:
    return sell_charges(1.0, cfg)


def apply_slippage(price: float, side: str, cfg: IndiaEquityCostConfig | None = None) -> float:
    """Buy fills higher and sell fills lower by `slippage_bps`."""
    if cfg is None:
        cfg = IndiaEquityCostConfig.equity_delivery()
    px = float(price)
    bump = float(cfg.slippage_bps) / 10000.0
    side_u = str(side).upper().strip()
    if side_u == "BUY":
        return px * (1.0 + bump)
    if side_u == "SELL":
        return px * (1.0 - bump)
    raise ValueError("side must be BUY or SELL")
