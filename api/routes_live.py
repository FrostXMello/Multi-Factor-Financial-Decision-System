"""Live Analysis API — scanner status, candidates, and manual scan triggers."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.deps import get_scan_store, get_scanner_service
from api.serialize import jsonable

router = APIRouter(prefix="/api/live", tags=["live-analysis"])


class ScanBody(BaseModel):
    symbols: Optional[list[str]] = None


@router.get("/status")
def live_status() -> dict:
    scanner = get_scanner_service()
    payload = scanner.status()
    payload["verified_live"] = bool(
        payload.get("latest_completed")
        and str((payload.get("latest_completed") or {}).get("data_mode", "")).upper() == "LIVE"
    )
    # Gate: UI should treat feed as unavailable-as-live until verified.
    if not payload["verified_live"]:
        payload["live_feed_state"] = "locked"
        payload["live_feed_detail"] = (
            "Verified LIVE tape is locked until a real provider passes market-session verification. "
            "Latest scanner results below use delayed/historical research data and are labelled accordingly."
        )
    else:
        payload["live_feed_state"] = "open"
        payload["live_feed_detail"] = "Market status reports verified LIVE quotes."
    return jsonable(payload)


@router.get("/candidates")
def live_candidates() -> dict:
    store = get_scan_store()
    run = store.latest_completed_run() or store.latest_run()
    if not run:
        return {
            "available": False,
            "detail": "No scan results stored yet. Trigger POST /api/live/scan or wait for the scheduler.",
            "broker_execution": False,
        }
    results = store.results_for_run(run["id"])
    buys = [r for r in results if r["signal"] == "BUY"]
    sells = [r for r in results if r["signal"] == "SELL"]
    return {
        "available": True,
        "run": run,
        "candidates": results,
        "strongest_buys": buys[:10],
        "strongest_sells": sells[:10],
        "broker_execution": False,
        "freshness": run.get("freshness"),
        "data_mode": run.get("data_mode"),
    }


@router.post("/scan")
def live_scan(body: Optional[ScanBody] = None) -> dict:
    scanner = get_scanner_service()
    try:
        result = scanner.run_once(trigger="manual", symbols=(body.symbols if body else None))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Scan failed: {exc}") from exc
    return jsonable(result)


@router.get("/history")
def live_history(limit: int = Query(default=10, ge=1, le=50)) -> dict:
    store = get_scan_store()
    # Lightweight: return latest run metadata only via repeated latest queries is weak;
    # expose candidates of latest completed for the dashboard history strip.
    run = store.latest_completed_run()
    if not run:
        return {"runs": [], "detail": "No completed scans yet."}
    return {
        "runs": [run],
        "candidates": store.results_for_run(run["id"])[:limit],
        "broker_execution": False,
    }
