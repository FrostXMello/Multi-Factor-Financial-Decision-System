"""
Continuous NSE universe scanner.

Uses the existing technical research pipeline. Never places real broker orders.
Verified LIVE mode is only claimed when market status data_mode is LIVE.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from live_analysis.store import ScanStore, open_scan_store


def _watchlist() -> list[str]:
    raw = os.environ.get("MARKET_DATA_WATCHLIST", "").strip()
    if raw:
        return [part.strip().upper() for part in raw.split(",") if part.strip()]
    # Default supported scanner universe subset (Yahoo NSE suffixes).
    return [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "LT.NS",
        "AXISBANK.NS",
    ]


def _interval_seconds() -> int:
    try:
        return max(60, int(os.environ.get("LIVE_SCAN_INTERVAL_S", "300")))
    except ValueError:
        return 300


def _enabled() -> bool:
    return os.environ.get("LIVE_SCANNER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _rank_score(signal: str, proba: float | None, no_trade: bool) -> float:
    p = float(proba) if proba is not None else 0.5
    sig = str(signal).upper()
    if no_trade and sig == "BUY":
        return p * 0.25
    if sig == "BUY":
        return p
    if sig == "SELL":
        return 1.0 - p
    return 0.35 + abs(p - 0.5)


def _reasons(latest: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = []
    signal = str(latest.get("signal") or "HOLD").upper()
    proba = latest.get("proba")
    reasons.append(
        {
            "factor": "model_probability",
            "detail": f"Calibrated/model P={proba}",
            "supports": signal if signal != "HOLD" else "neutral",
        }
    )
    if latest.get("ma50") is not None and latest.get("close") is not None:
        above = float(latest["close"]) >= float(latest["ma50"])
        reasons.append(
            {
                "factor": "ma50_trend",
                "detail": f"Close vs MA50: {'above' if above else 'below'}",
                "supports": "BUY" if above else "SELL",
            }
        )
    if latest.get("no_trade"):
        reasons.append(
            {
                "factor": "risk_engine",
                "detail": "NoTrade gate blocks new BUYs; exits remain allowed.",
                "supports": "invalidation",
            }
        )
    if latest.get("risk_score") is not None:
        reasons.append(
            {
                "factor": "risk_score",
                "detail": f"RiskScore={latest.get('risk_score')}; PositionSize={latest.get('position_size')}",
                "supports": "risk",
            }
        )
    if metrics.get("walk_forward_mean_accuracy") is not None:
        reasons.append(
            {
                "factor": "walk_forward",
                "detail": f"OOS mean accuracy={metrics.get('walk_forward_mean_accuracy')} (not P(profit))",
                "supports": "uncertainty",
            }
        )
    return reasons


class MarketScanner:
    def __init__(self, store: ScanStore):
        self.store = store
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running_scan = False
        self._last_error = ""
        self._next_scan_at: datetime | None = None
        self._status = "idle"

    def status(self) -> dict[str, Any]:
        with self._lock:
            latest = self.store.latest_run()
            completed = self.store.latest_completed_run()
            results = self.store.results_for_run(completed["id"]) if completed else []
            return {
                "scanner_enabled": _enabled(),
                "scanner_state": self._status,
                "running_scan": self._running_scan,
                "last_error": self._last_error,
                "interval_seconds": _interval_seconds(),
                "next_scan_at": self._next_scan_at.isoformat() if self._next_scan_at else (latest or {}).get("next_scan_at"),
                "universe": _watchlist(),
                "broker_execution": False,
                "latest_run": latest,
                "latest_completed": completed,
                "candidates": results,
                "note": (
                    "Scanner ranks existing model outputs on delayed/historical research data unless "
                    "market status reports verified LIVE. It never places real-money orders."
                ),
            }

    def start_background(self) -> None:
        if not _enabled():
            self._status = "disabled"
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="market-scanner", daemon=True)
            self._thread.start()
            self._status = "running"

    def stop_background(self) -> None:
        self._stop.set()
        self._status = "stopped"

    def _loop(self) -> None:
        # Stagger first scan slightly so API boot completes.
        self._stop.wait(5)
        while not self._stop.is_set():
            session = self._session_status()
            should = session == "open" or os.environ.get("LIVE_SCAN_WHEN_CLOSED", "0").strip() in {"1", "true"}
            if should and not self._running_scan:
                try:
                    self.run_once(trigger="schedule")
                except Exception as exc:  # noqa: BLE001
                    self._last_error = str(exc)
                    self._status = "error"
            wait = _interval_seconds()
            self._next_scan_at = datetime.now(timezone.utc) + timedelta(seconds=wait)
            self._stop.wait(wait)

    def _session_status(self) -> str:
        from market_data.session import nse_cash_status

        return nse_cash_status(datetime.now(timezone.utc))

    def _market_meta(self) -> dict[str, Any]:
        from market_data.providers import BrokerLiveProvider, YFinanceHistoryProvider
        from market_data.session import nse_cash_status

        now = datetime.now(timezone.utc)
        session = nse_cash_status(now)
        yahoo = YFinanceHistoryProvider().status()
        broker = BrokerLiveProvider().status()
        # Honest mode: verified LIVE only when open + broker live quotes exist.
        # Without that, scanner is SIMULATED/RESEARCH over delayed history.
        if session == "open" and broker.configured and str(broker.freshness).lower() == "live":
            data_mode = "LIVE"
            freshness = "live"
            source = "broker_live"
        elif session != "open":
            data_mode = "CLOSED"
            freshness = "historical"
            source = "yfinance"
        else:
            data_mode = "DELAYED"
            freshness = "delayed"
            source = "yfinance"
        return {
            "session_status": session,
            "data_mode": data_mode,
            "freshness": freshness,
            "source": source,
            "yahoo_freshness": yahoo.freshness,
            "broker_configured": broker.configured,
            "live_verified": data_mode == "LIVE",
        }

    def run_once(self, *, trigger: str = "manual", symbols: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._running_scan:
                return {"status": "busy", "detail": "A scan is already running."}
            self._running_scan = True
            self._status = "scanning"
        try:
            return self._run_scan(trigger=trigger, symbols=symbols)
        finally:
            with self._lock:
                self._running_scan = False
                if self._status == "scanning":
                    self._status = "running" if _enabled() else "idle"

    def _run_scan(self, *, trigger: str, symbols: list[str] | None) -> dict[str, Any]:
        from api.serialize import summarize_research
        from four_model_pipeline import FourModelConfig, run_four_model_pipeline

        meta = self._market_meta()
        universe = [s.strip().upper() for s in (symbols or _watchlist()) if s.strip()]
        next_at = (datetime.now(timezone.utc) + timedelta(seconds=_interval_seconds())).isoformat()
        run_id = self.store.start_run(
            session_status=meta["session_status"],
            data_mode=meta["data_mode"],
            freshness=meta["freshness"],
            source=f"{meta['source']}:{trigger}",
            universe=universe,
            next_scan_at=next_at,
        )
        failures = 0
        for symbol in universe:
            try:
                result = run_four_model_pipeline(symbol, FourModelConfig())
                summary = summarize_research({**result, "ticker": symbol})
                latest = dict(summary.get("latest") or {})
                metrics = dict(summary.get("metrics") or {})
                signal = str(latest.get("signal") or "HOLD").upper()
                proba = latest.get("proba")
                no_trade = bool(latest.get("no_trade"))
                explanation = {}
                try:
                    from reasoning.service import ExplainRequest, explain_decision

                    explanation = explain_decision(
                        ExplainRequest(
                            ticker=symbol,
                            latest=latest,
                            reliability=dict(summary.get("reliability") or {}),
                            metrics=metrics,
                            strategy_mode=summary.get("strategy_mode"),
                            buy_threshold=(summary.get("thresholds") or {}).get("buy_threshold", 0.6),
                            sell_threshold=(summary.get("thresholds") or {}).get("sell_threshold", 0.4),
                            use_llm=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    explanation = {"summary": f"Explanation unavailable: {exc}", "mode": "error"}

                self.store.add_result(
                    run_id,
                    {
                        "symbol": symbol,
                        "signal": signal,
                        "proba": proba,
                        "rank_score": _rank_score(signal, float(proba) if proba is not None else None, no_trade),
                        "risk_score": latest.get("risk_score"),
                        "position_size": latest.get("position_size"),
                        "no_trade": no_trade,
                        "close": latest.get("close"),
                        "reasons": _reasons(latest, metrics),
                        "metrics": metrics,
                        "explanation": explanation,
                        "detail": "",
                    },
                )
                # Soft rate limit between Yahoo-backed research pulls.
                time.sleep(0.35)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                self._last_error = f"{symbol}: {exc}"
                self.store.add_result(
                    run_id,
                    {
                        "symbol": symbol,
                        "signal": "HOLD",
                        "proba": None,
                        "rank_score": -1.0,
                        "risk_score": None,
                        "position_size": None,
                        "no_trade": True,
                        "close": None,
                        "reasons": [{"factor": "provider_failure", "detail": str(exc), "supports": "invalidation"}],
                        "metrics": {},
                        "explanation": {},
                        "detail": str(exc),
                    },
                )

        status = "completed" if failures < len(universe) else "failed"
        error = self._last_error if failures else ""
        self.store.finish_run(run_id, status=status, error=error, next_scan_at=next_at)
        self._next_scan_at = datetime.now(timezone.utc) + timedelta(seconds=_interval_seconds())
        completed = self.store.latest_completed_run() or self.store.latest_run()
        return {
            "run": completed,
            "candidates": self.store.results_for_run(run_id),
            "meta": meta,
            "broker_execution": False,
            "failures": failures,
        }


_SCANNER: MarketScanner | None = None
_SCANNER_LOCK = threading.Lock()


def get_scanner(store_factory: Callable[[], ScanStore] | None = None) -> MarketScanner:
    global _SCANNER
    with _SCANNER_LOCK:
        if _SCANNER is None:
            store = store_factory() if store_factory else open_scan_store()
            _SCANNER = MarketScanner(store)
        return _SCANNER


def clear_scanner() -> None:
    global _SCANNER
    with _SCANNER_LOCK:
        if _SCANNER is not None:
            _SCANNER.stop_background()
        _SCANNER = None
