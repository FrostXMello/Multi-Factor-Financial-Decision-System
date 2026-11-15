"""
SQLite persistence for paper and demo accounts.

Demo rows use account_id values that start with `demo:`. They never share an
account_id with the normal paper book.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    label TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    cash REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    amount REAL NOT NULL,
    balance_after REAL NOT NULL,
    ref TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE INDEX IF NOT EXISTS idx_ledger_account_ts ON ledger (account_id, ts);

CREATE TABLE IF NOT EXISTS positions (
    account_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_price REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (account_id, symbol),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    filled_at TEXT,
    requested_price REAL,
    fill_price REAL,
    fees REAL NOT NULL DEFAULT 0,
    price_source TEXT NOT NULL DEFAULT '',
    price_time TEXT,
    freshness TEXT NOT NULL DEFAULT '',
    rejection_reason TEXT NOT NULL DEFAULT '',
    signal_ref TEXT NOT NULL DEFAULT '',
    UNIQUE (account_id, client_order_id),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE INDEX IF NOT EXISTS idx_orders_account_ts ON orders (account_id, submitted_at);

CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT NOT NULL,
    realized_pnl REAL NOT NULL,
    fees REAL NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE INDEX IF NOT EXISTS idx_closed_account ON closed_trades (account_id, exit_ts);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    drawdown REAL NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE INDEX IF NOT EXISTS idx_equity_account_ts ON equity_curve (account_id, ts);
"""


def default_paper_db_path() -> Path:
    raw = os.environ.get("PAPER_TRADING_DB")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data" / "paper.sqlite"


def open_paper_store(path: Path | str | None = None) -> "PaperStore":
    return PaperStore(path or default_paper_db_path())


class PaperStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        cur.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        current = int(row["version"]) if row else 0
        if current < 1:
            self.conn.executescript(_SCHEMA_V1)
            cur.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (1, _utc_now_iso()),
            )
            self.conn.commit()

    def ensure_account(
        self,
        account_id: str,
        *,
        mode: str,
        label: str,
        initial_capital: float,
    ) -> dict:
        mode_u = str(mode).strip().lower()
        if mode_u not in {"paper", "demo"}:
            raise ValueError("mode must be paper or demo")
        if mode_u == "demo" and not str(account_id).startswith("demo:"):
            raise ValueError("demo accounts must use an account_id starting with 'demo:'")
        if mode_u == "paper" and str(account_id).startswith("demo:"):
            raise ValueError("paper accounts must not use the demo: prefix")
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        existing = cur.fetchone()
        if existing:
            return dict(existing)
        now = _utc_now_iso()
        capital = float(initial_capital)
        cur.execute(
            """
            INSERT INTO accounts(account_id, mode, label, initial_capital, cash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (account_id, mode_u, label, capital, capital, now, now),
        )
        cur.execute(
            """
            INSERT INTO ledger(account_id, ts, kind, amount, balance_after, ref, detail)
            VALUES (?, ?, 'deposit', ?, ?, '', 'Initial virtual capital')
            """,
            (account_id, now, capital, capital),
        )
        self.conn.commit()
        cur.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        return dict(cur.fetchone())

    def reset_account(self, account_id: str, *, initial_capital: float | None = None) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(account_id)
        capital = float(initial_capital if initial_capital is not None else row["initial_capital"])
        now = _utc_now_iso()
        for table in ("ledger", "positions", "orders", "closed_trades", "equity_curve"):
            cur.execute(f"DELETE FROM {table} WHERE account_id = ?", (account_id,))
        cur.execute(
            """
            UPDATE accounts
            SET initial_capital = ?, cash = ?, updated_at = ?
            WHERE account_id = ?
            """,
            (capital, capital, now, account_id),
        )
        cur.execute(
            """
            INSERT INTO ledger(account_id, ts, kind, amount, balance_after, ref, detail)
            VALUES (?, ?, 'deposit', ?, ?, '', 'Reset virtual capital')
            """,
            (account_id, now, capital, capital),
        )
        self.conn.commit()
        cur.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        return dict(cur.fetchone())

    def get_account(self, account_id: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def set_cash(self, account_id: str, cash: float, *, kind: str, amount: float, ref: str = "", detail: str = "") -> None:
        now = _utc_now_iso()
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE accounts SET cash = ?, updated_at = ? WHERE account_id = ?",
            (float(cash), now, account_id),
        )
        cur.execute(
            """
            INSERT INTO ledger(account_id, ts, kind, amount, balance_after, ref, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (account_id, now, kind, float(amount), float(cash), ref, detail),
        )
        self.conn.commit()

    def get_position(self, account_id: str, symbol: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM positions WHERE account_id = ? AND symbol = ?",
            (account_id, symbol.upper()),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_positions(self, account_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM positions WHERE account_id = ? ORDER BY symbol", (account_id,))
        return [dict(row) for row in cur.fetchall()]

    def upsert_position(self, account_id: str, symbol: str, quantity: float, avg_price: float) -> None:
        symbol_u = symbol.upper()
        now = _utc_now_iso()
        cur = self.conn.cursor()
        if float(quantity) <= 1e-12:
            cur.execute("DELETE FROM positions WHERE account_id = ? AND symbol = ?", (account_id, symbol_u))
        else:
            cur.execute(
                """
                INSERT INTO positions(account_id, symbol, quantity, avg_price, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_id, symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_price = excluded.avg_price,
                    updated_at = excluded.updated_at
                """,
                (account_id, symbol_u, float(quantity), float(avg_price), now),
            )
        self.conn.commit()

    def insert_order(self, order: dict) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO orders(
                order_id, account_id, client_order_id, symbol, side, quantity, status,
                submitted_at, filled_at, requested_price, fill_price, fees,
                price_source, price_time, freshness, rejection_reason, signal_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order["order_id"],
                order["account_id"],
                order["client_order_id"],
                order["symbol"],
                order["side"],
                order["quantity"],
                order["status"],
                order["submitted_at"],
                order.get("filled_at"),
                order.get("requested_price"),
                order.get("fill_price"),
                order.get("fees", 0.0),
                order.get("price_source", ""),
                order.get("price_time"),
                order.get("freshness", ""),
                order.get("rejection_reason", ""),
                order.get("signal_ref", ""),
            ),
        )
        self.conn.commit()

    def find_order_by_client_id(self, account_id: str, client_order_id: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM orders WHERE account_id = ? AND client_order_id = ?",
            (account_id, client_order_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def recent_orders(self, account_id: str, limit: int = 50) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM orders
            WHERE account_id = ?
            ORDER BY submitted_at DESC, order_id DESC
            LIMIT ?
            """,
            (account_id, int(limit)),
        )
        return [dict(row) for row in cur.fetchall()]

    def insert_closed_trade(self, trade: dict) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO closed_trades(
                account_id, symbol, quantity, entry_price, exit_price, entry_ts, exit_ts, realized_pnl, fees
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade["account_id"],
                trade["symbol"],
                trade["quantity"],
                trade["entry_price"],
                trade["exit_price"],
                trade["entry_ts"],
                trade["exit_ts"],
                trade["realized_pnl"],
                trade["fees"],
            ),
        )
        self.conn.commit()

    def closed_trades(self, account_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM closed_trades WHERE account_id = ? ORDER BY exit_ts ASC, id ASC",
            (account_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def realized_pnl_today(self, account_id: str, day_prefix: str) -> float:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0) AS pnl
            FROM closed_trades
            WHERE account_id = ? AND exit_ts LIKE ?
            """,
            (account_id, f"{day_prefix}%"),
        )
        return float(cur.fetchone()["pnl"])

    def append_equity(self, account_id: str, *, cash: float, equity: float, drawdown: float) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO equity_curve(account_id, ts, cash, equity, drawdown)
            VALUES (?, ?, ?, ?, ?)
            """,
            (account_id, _utc_now_iso(), float(cash), float(equity), float(drawdown)),
        )
        self.conn.commit()

    def equity_points(self, account_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM equity_curve WHERE account_id = ? ORDER BY ts ASC, id ASC",
            (account_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def stats(self) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(version) AS v FROM schema_migrations")
        version = cur.fetchone()["v"] or 0
        counts = {}
        for table in ("accounts", "orders", "positions", "closed_trades", "ledger"):
            cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(cur.fetchone()["n"])
        return {"path": str(self.path), "schema_version": int(version), **counts}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
