"""SQLite persistence for live scans and saved backtest runs."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    session_status TEXT NOT NULL,
    data_mode TEXT NOT NULL,
    freshness TEXT NOT NULL,
    source TEXT NOT NULL,
    universe_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    next_scan_at TEXT,
    broker_execution INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scan_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    signal TEXT NOT NULL,
    proba REAL,
    rank_score REAL,
    risk_score REAL,
    position_size REAL,
    no_trade INTEGER NOT NULL DEFAULT 0,
    close REAL,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    explanation_json TEXT NOT NULL DEFAULT '{}',
    detail TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(run_id) REFERENCES scan_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_scan_results_run ON scan_results(run_id);
CREATE INDEX IF NOT EXISTS idx_scan_results_symbol ON scan_results(symbol);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    freshness TEXT NOT NULL DEFAULT 'historical'
);
"""


def default_scan_db_path() -> Path:
    raw = os.environ.get("SCAN_DB", "").strip()
    if raw:
        return Path(raw).expanduser()
    data_dir = os.environ.get("DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser() / "scans.sqlite"
    return Path(__file__).resolve().parents[1] / "data" / "scans.sqlite"


def open_scan_store(path: str | Path | None = None) -> "ScanStore":
    return ScanStore(Path(path) if path is not None else default_scan_db_path())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ScanStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.migrate()

    def close(self) -> None:
        self._conn.close()

    def migrate(self) -> int:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        current = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
        if int(current) < 1:
            self._conn.executescript(_SCHEMA_V1)
            self._conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (1, _utc_now()),
            )
            self._conn.commit()
        return int(self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] or 0)

    def start_run(
        self,
        *,
        session_status: str,
        data_mode: str,
        freshness: str,
        source: str,
        universe: list[str],
        next_scan_at: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO scan_runs (
                started_at, session_status, data_mode, freshness, source,
                universe_json, status, next_scan_at, broker_execution
            ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, 0)
            """,
            (
                _utc_now(),
                session_status,
                data_mode,
                freshness,
                source,
                json.dumps(universe),
                next_scan_at,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, status: str, error: str = "", next_scan_at: str | None = None) -> None:
        self._conn.execute(
            """
            UPDATE scan_runs
            SET finished_at = ?, status = ?, error = ?, next_scan_at = COALESCE(?, next_scan_at)
            WHERE id = ?
            """,
            (_utc_now(), status, error, next_scan_at, run_id),
        )
        self._conn.commit()

    def add_result(self, run_id: int, payload: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO scan_results (
                run_id, symbol, signal, proba, rank_score, risk_score, position_size,
                no_trade, close, reasons_json, metrics_json, explanation_json, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                str(payload.get("symbol", "")).upper(),
                str(payload.get("signal", "HOLD")).upper(),
                payload.get("proba"),
                payload.get("rank_score"),
                payload.get("risk_score"),
                payload.get("position_size"),
                1 if payload.get("no_trade") else 0,
                payload.get("close"),
                json.dumps(payload.get("reasons") or []),
                json.dumps(payload.get("metrics") or {}),
                json.dumps(payload.get("explanation") or {}),
                str(payload.get("detail") or ""),
            ),
        )
        self._conn.commit()

    def latest_run(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM scan_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._run_dict(row) if row else None

    def latest_completed_run(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM scan_runs WHERE status = 'completed' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._run_dict(row) if row else None

    def results_for_run(self, run_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM scan_results WHERE run_id = ? ORDER BY rank_score DESC, symbol ASC",
            (run_id,),
        ).fetchall()
        return [self._result_dict(row) for row in rows]

    def save_backtest(self, *, ticker: str, label: str, params: dict, summary: dict, freshness: str) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO backtest_runs (created_at, ticker, label, params_json, summary_json, freshness)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_utc_now(), ticker.upper(), label, json.dumps(params), json.dumps(summary), freshness),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_backtests(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, created_at, ticker, label, freshness, params_json FROM backtest_runs ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 100)),),
        ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "ticker": row["ticker"],
                    "label": row["label"],
                    "freshness": row["freshness"],
                    "params": json.loads(row["params_json"] or "{}"),
                }
            )
        return out

    def get_backtest(self, run_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "ticker": row["ticker"],
            "label": row["label"],
            "freshness": row["freshness"],
            "params": json.loads(row["params_json"] or "{}"),
            "summary": json.loads(row["summary_json"] or "{}"),
        }

    def _run_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "session_status": row["session_status"],
            "data_mode": row["data_mode"],
            "freshness": row["freshness"],
            "source": row["source"],
            "universe": json.loads(row["universe_json"] or "[]"),
            "status": row["status"],
            "error": row["error"],
            "next_scan_at": row["next_scan_at"],
            "broker_execution": bool(row["broker_execution"]),
        }

    def _result_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "symbol": row["symbol"],
            "signal": row["signal"],
            "proba": row["proba"],
            "rank_score": row["rank_score"],
            "risk_score": row["risk_score"],
            "position_size": row["position_size"],
            "no_trade": bool(row["no_trade"]),
            "close": row["close"],
            "reasons": json.loads(row["reasons_json"] or "[]"),
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "explanation": json.loads(row["explanation_json"] or "{}"),
            "detail": row["detail"],
        }
