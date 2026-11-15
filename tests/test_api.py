"""API tests use temporary SQLite files and never call a live broker."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_DB", str(tmp_path / "market.sqlite"))
    monkeypatch.setenv("PAPER_TRADING_DB", str(tmp_path / "paper.sqlite"))
    monkeypatch.setenv("MARKET_DATA_WATCHLIST", "RELIANCE.NS,TCS.NS")
    monkeypatch.delenv("BROKER_LIVE_BASE_URL", raising=False)
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_ACCESS_TOKEN", raising=False)

    from api.deps import clear_store_caches
    from api.routes_research import clear_research_cache

    clear_store_caches()
    clear_research_cache()

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    clear_store_caches()
    clear_research_cache()


def test_health_and_overview_disable_broker_execution(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["broker_execution"] is False
    overview = client.get("/api/overview").json()
    assert "nse_cash_session" in overview
    assert overview["broker_configured"] is False
    assert "disclaimer" in overview


def test_market_status_and_closed_collect(client):
    status = client.get("/api/market/status").json()
    assert status["data_mode"] in {"HISTORICAL", "UNCONFIGURED", "CLOSED", "DELAYED"}
    assert any(p["provider_id"] == "yfinance" for p in status["providers"])
    broker = next(p for p in status["providers"] if p["provider_id"] == "broker_live")
    assert broker["configured"] is False
    collect = client.post("/api/market/collect")
    assert collect.status_code == 200
    body = collect.json()
    assert body["fetched"] == 0
    assert body["freshness"] == "unavailable"


def test_paper_demo_isolation_via_api(client):
    demo = client.post("/api/paper/demo/run")
    assert demo.status_code == 200
    payload = demo.json()
    assert payload["demo_labelled"] is True
    assert payload["account_id"].startswith("demo:")
    assert all(item["accepted"] for item in payload["results"])
    assert all(item["order"]["freshness"] == "demo" for item in payload["results"])

    paper = client.get("/api/paper/account", params={"mode": "paper"}).json()
    assert paper["account_id"] == "paper:default"
    assert paper["snapshot"]["cash"] == pytest.approx(100_000.0)
    assert paper["snapshot"]["positions"] == []

    rejected = client.post(
        "/api/paper/orders",
        json={
            "symbol": "RELIANCE.NS",
            "side": "BUY",
            "quantity": 1,
            "mode": "paper",
            "confirm": True,
            "allow_historical": True,
        },
    )
    # No stored market mark in the temp DB → order rejected, not fabricated live.
    assert rejected.status_code == 400


def test_research_last_empty_and_run_monkeypatched(client, monkeypatch):
    empty = client.get("/api/research/last").json()
    assert empty["available"] is False

    import pandas as pd

    fake = {
        "ticker": "RELIANCE.NS",
        "strategy_mode": "technical_only",
        "technical_test_accuracy": 0.6,
        "walk_forward_mean_accuracy": 0.55,
        "walk_forward_fold_scores": [0.5, 0.6],
        "trades": 2,
        "trades_per_day": 0.1,
        "test_df": pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "Close": [100.0, 101.0],
                "Signal": ["HOLD", "BUY"],
                "Proba": [0.4, 0.7],
                "RiskScore": [0.1, 0.2],
                "PositionSize": [1.0, 1.0],
            }
        ),
        "backtest": {
            "final_portfolio_value": 1010.0,
            "total_return_pct": 1.0,
            "gross_return_pct": 1.2,
            "total_transaction_costs": 2.0,
            "max_drawdown": -0.01,
            "sharpe": 0.5,
            "equity_curve": pd.Series([1000.0, 1010.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"])),
            "trades_df": pd.DataFrame(),
        },
        "reliability": {
            "evaluation": "walk_forward",
            "decision_method": "sigmoid",
            "insample_accuracy": 0.7,
            "oos_metrics": {"accuracy": 0.55},
        },
    }
    monkeypatch.setattr("api.routes_research.run_four_model_pipeline", lambda ticker, cfg=None: fake)
    run = client.post("/api/research/run", json={"ticker": "RELIANCE.NS"})
    assert run.status_code == 200
    body = run.json()
    assert body["latest"]["signal"] == "BUY"
    assert body["reliability"]["evaluation"] == "walk_forward"
    last = client.get("/api/research/last").json()
    assert last["available"] is True
    rel = client.get("/api/research/reliability").json()
    assert rel["available"] is True
