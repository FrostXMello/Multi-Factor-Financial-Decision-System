"""Deployment readiness smoke checks (offline TestClient; no hosting credentials required)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("MARKET_DATA_DB", raising=False)
    monkeypatch.delenv("PAPER_TRADING_DB", raising=False)
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


def test_health_ready_and_persistence_defaults(client, tmp_path):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["broker_execution"] is False
    assert "live_feed_verification" in health

    ready = client.get("/api/health/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["broker_execution"] is False
    assert str(tmp_path / "data") in body["persistence"]["data_dir"]
    assert body["persistence"]["market_data_db"].endswith("market.sqlite")
    assert body["persistence"]["paper_trading_db"].endswith("paper.sqlite")


def test_diagnostics_never_exposes_secrets(client, monkeypatch):
    monkeypatch.setenv("BROKER_API_KEY", "super-secret-broker")
    monkeypatch.setenv("AI_REASONING_API_KEY", "super-secret-ai")
    monkeypatch.setenv("BROKER_LIVE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "tok")
    diag = client.get("/api/diagnostics").json()
    blob = json.dumps(diag)
    assert "super-secret-broker" not in blob
    assert "super-secret-ai" not in blob
    assert diag["broker_execution_enabled"] is False
    assert diag["secrets_exposed"] is False
    assert diag["runtime"]["broker_execution"] is False


def test_paper_demo_isolation_and_no_live_orders(client):
    demo = client.post("/api/paper/demo/run").json()
    assert demo["demo_labelled"] is True
    assert demo["account_id"].startswith("demo:")
    paper = client.get("/api/paper/account", params={"mode": "paper"}).json()
    assert paper["account_id"] == "paper:default"
    assert paper["snapshot"]["cash"] == pytest.approx(100_000.0)
    assert paper["snapshot"]["positions"] == []
    assert client.get("/api/health").json()["broker_execution"] is False


def test_market_status_freshness_and_closed_collect(client):
    status = client.get("/api/market/status").json()
    assert status["data_mode"] in {"HISTORICAL", "UNCONFIGURED", "CLOSED", "DELAYED", "LIVE"}
    assert "pending verification" in status["note"].lower() or "pending" in status["note"].lower()
    yahoo = next(p for p in status["providers"] if p["provider_id"] == "yfinance")
    assert yahoo["freshness"] == "delayed"
    broker = next(p for p in status["providers"] if p["provider_id"] == "broker_live")
    assert broker["configured"] is False
    collect = client.post("/api/market/collect").json()
    assert collect["fetched"] == 0
