"""Live scanner + backtest API smoke tests (offline, no live broker)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MARKET_DATA_DB", str(tmp_path / "market.sqlite"))
    monkeypatch.setenv("PAPER_TRADING_DB", str(tmp_path / "paper.sqlite"))
    monkeypatch.setenv("SCAN_DB", str(tmp_path / "scans.sqlite"))
    monkeypatch.setenv("MARKET_DATA_WATCHLIST", "RELIANCE.NS")
    monkeypatch.setenv("LIVE_SCANNER_ENABLED", "0")
    monkeypatch.setenv("LIVE_SCAN_WHEN_CLOSED", "0")
    monkeypatch.delenv("BROKER_LIVE_BASE_URL", raising=False)
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("AI_REASONING_API_KEY", raising=False)

    from api.deps import clear_store_caches
    from api.routes_analytics import clear_analytics_cache
    from api.routes_explain import clear_explain_cache
    from api.routes_research import clear_research_cache

    clear_store_caches()
    clear_research_cache()
    clear_explain_cache()
    clear_analytics_cache()

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    clear_store_caches()


def test_live_status_locked_and_no_broker_execution(client):
    status = client.get("/api/live/status").json()
    assert status["broker_execution"] is False
    assert status["live_feed_state"] == "locked"
    assert status["scanner_enabled"] is False
    candidates = client.get("/api/live/candidates").json()
    assert candidates["broker_execution"] is False


def test_backtest_defaults_and_symbols(client):
    defaults = client.get("/api/backtest/defaults").json()
    assert defaults["freshness"] == "historical"
    assert "buy_threshold" in defaults
    assert "costs" in defaults
    assert len(defaults["strategy_presets"]) == 5
    symbols = client.get("/api/backtest/symbols").json()
    assert "symbols" in symbols
    assert "RELIANCE.NS" in symbols["symbols"]
    assert symbols["freshness"] == "historical"


def test_simulate_strategies_ranks_five_and_explains(client, monkeypatch):
    from api import routes_backtest as rb

    def fake_run(ticker, cfg, start=None, end=None):
        mode = getattr(cfg, "strategy_mode", "technical_only")
        return {
            "ticker": ticker,
            "strategy_mode": mode,
            "test_df": __import__("pandas").DataFrame(
                [
                    {
                        "Date": "2024-01-02",
                        "Close": 100.0,
                        "MA50": 95.0,
                        "Signal": "BUY",
                        "Proba": 0.72,
                        "RiskScore": 0.2,
                        "PositionSize": 0.8,
                        "NoTrade": False,
                    }
                ]
            ),
            "backtest": {
                "final_portfolio_value": 110000.0,
                "total_return_pct": 10.0,
                "gross_return_pct": 11.0,
                "total_transaction_costs": 100.0,
                "max_drawdown_pct": -5.0,
                "sharpe": 1.2,
                "cagr_pct": 8.0,
                "win_rate": 0.55,
                "trades_df": __import__("pandas").DataFrame(),
            },
            "technical_test_accuracy": 0.6,
            "walk_forward_mean_accuracy": 0.58,
            "walk_forward_fold_scores": [0.5, 0.6],
            "trades": 4,
            "trades_per_day": 0.1,
            "reliability": {"note": "test"},
            "config": {"buy_threshold": cfg.buy_threshold, "sell_threshold": cfg.sell_threshold},
        }

    monkeypatch.setattr(rb, "run_four_model_pipeline", fake_run)
    missing = client.post(
        "/api/backtest/simulate-strategies",
        json={"ticker": "RELIANCE.NS", "save": False},
    )
    assert missing.status_code == 422

    body = client.post(
        "/api/backtest/simulate-strategies",
        json={
            "ticker": "RELIANCE.NS",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "save": False,
        },
    ).json()
    assert body["broker_execution"] is False
    assert body["freshness"] == "historical"
    assert body["window"]["blocked_outside_window"] is True
    assert len(body["strategies"]) == 5
    assert body["strategies"][0]["rank"] == 1
    assert body["buy_analysis"]["available"] is True
    assert body["buy_analysis"]["latest_signal"] == "BUY"
    assert "why_buy" in body["buy_analysis"]
    assert "orders" in body["strategies"][0]


def test_scan_store_persists_manual_result(client, monkeypatch):
    """Persist a synthetic scan without calling Yahoo."""
    from api.deps import get_scan_store

    store = get_scan_store()
    run_id = store.start_run(
        session_status="closed",
        data_mode="CLOSED",
        freshness="historical",
        source="test",
        universe=["RELIANCE.NS"],
    )
    store.add_result(
        run_id,
        {
            "symbol": "RELIANCE.NS",
            "signal": "BUY",
            "proba": 0.71,
            "rank_score": 0.71,
            "risk_score": 0.2,
            "position_size": 0.8,
            "no_trade": False,
            "close": 100.0,
            "reasons": [{"factor": "test", "detail": "unit", "supports": "BUY"}],
            "metrics": {"net_return_pct": 1.2},
            "explanation": {"summary": "test explanation"},
        },
    )
    store.finish_run(run_id, status="completed")
    body = client.get("/api/live/candidates").json()
    assert body["available"] is True
    assert body["candidates"][0]["symbol"] == "RELIANCE.NS"
    assert body["candidates"][0]["signal"] == "BUY"
    assert body["freshness"] == "historical"


def test_save_and_compare_backtest_runs(client):
    from api.deps import get_scan_store

    store = get_scan_store()
    a = store.save_backtest(
        ticker="RELIANCE.NS",
        label="A",
        params={"buy_threshold": 0.6},
        summary={"metrics": {"net_return_pct": 1.0, "sharpe": 0.5}},
        freshness="historical",
    )
    b = store.save_backtest(
        ticker="TCS.NS",
        label="B",
        params={"buy_threshold": 0.65},
        summary={"metrics": {"net_return_pct": -1.0, "sharpe": 0.1}},
        freshness="historical",
    )
    listed = client.get("/api/backtest/runs").json()
    assert len(listed["runs"]) >= 2
    cmp_body = client.post("/api/backtest/compare", json={"run_ids": [a, b]}).json()
    assert len(cmp_body["comparisons"]) == 2
    assert "does not search parameters" in cmp_body["note"].lower() or "test folds" in cmp_body["note"].lower()
