"""
SQLite storage for instruments, candles, quotes, and quality events.

The public methods are the storage interface. A Postgres implementation can keep
the same method names later. This module does not open a network connection.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market_data.quality import find_price_jumps, ohlc_problems
from market_data.types import Instrument, Quote

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS providers (
    provider_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    description TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS instruments (
    symbol TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    asset_class TEXT NOT NULL DEFAULT 'equity',
    currency TEXT NOT NULL DEFAULT 'INR',
    lot_size INTEGER,
    source TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    provider_id TEXT NOT NULL,
    adjusted INTEGER NOT NULL DEFAULT 1,
    quality_flags TEXT NOT NULL DEFAULT '',
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (symbol, timeframe, bar_time);

CREATE TABLE IF NOT EXISTS quotes (
    symbol TEXT NOT NULL,
    quote_time TEXT NOT NULL,
    last_price REAL NOT NULL,
    bid REAL,
    ask REAL,
    volume REAL,
    provider_id TEXT NOT NULL,
    freshness TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (symbol, quote_time, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time
    ON quotes (symbol, quote_time);

CREATE TABLE IF NOT EXISTS quality_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timeframe TEXT,
    bar_time TEXT,
    check_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail TEXT NOT NULL,
    provider_id TEXT,
    detected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quality_detected
    ON quality_events (detected_at);

CREATE TABLE IF NOT EXISTS provider_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    raw = os.environ.get("MARKET_DATA_DB", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "market.sqlite"


def open_store(path: str | Path | None = None) -> "MarketStore":
    target = Path(path) if path is not None else default_db_path()
    return MarketStore(target)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _exchange_for(symbol: str) -> str:
    upper = symbol.upper()
    if upper.endswith(".NS"):
        return "NSE"
    if upper.endswith(".BO"):
        return "BSE"
    return "UNKNOWN"


class MarketStore:
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
        return int(self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])

    def register_provider(self, provider_id: str, kind: str, description: str) -> None:
        now = _utc_now()
        self._conn.execute(
            """
            INSERT INTO providers (provider_id, kind, description, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider_id) DO UPDATE SET
                kind = excluded.kind,
                description = excluded.description,
                updated_at = excluded.updated_at
            """,
            (provider_id, kind, description, now),
        )
        self._conn.commit()

    def upsert_instrument(self, instrument: Instrument) -> None:
        self._conn.execute(
            """
            INSERT INTO instruments (
                symbol, exchange, name, asset_class, currency, lot_size, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                exchange = excluded.exchange,
                name = excluded.name,
                asset_class = excluded.asset_class,
                currency = excluded.currency,
                lot_size = excluded.lot_size,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                instrument.symbol.strip().upper(),
                instrument.exchange,
                instrument.name,
                instrument.asset_class,
                instrument.currency,
                instrument.lot_size,
                instrument.source,
                _utc_now(),
            ),
        )
        self._conn.commit()

    def upsert_candles(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        provider_id: str,
        adjusted: bool = True,
        jump_threshold: float = 0.25,
    ) -> dict:
        """
        Insert or update candles. Invalid OHLC rows are skipped and recorded.
        Repeating the same bar updates it in place; the primary key blocks duplicates.
        """
        symbol = symbol.strip().upper()
        timeframe = str(timeframe).strip().lower()
        if frame is None or frame.empty:
            return {"written": 0, "rejected": 0}
        self.upsert_instrument(Instrument(symbol=symbol, exchange=_exchange_for(symbol), source=provider_id))
        ordered = frame.copy()
        ordered["Date"] = pd.to_datetime(ordered["Date"])
        ordered = ordered.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        prior = self.read_candles(symbol, timeframe, adjusted=adjusted)
        if len(prior):
            boundary = pd.concat(
                [prior.tail(1)[["Date", "Open", "High", "Low", "Close", "Volume"]], ordered],
                ignore_index=True,
            )
        else:
            boundary = ordered
        jumps = {pd.Timestamp(item["Date"]) for item in find_price_jumps(boundary, threshold=jump_threshold)}
        now = _utc_now()
        written = 0
        rejected = 0
        rows = []
        for record in ordered.itertuples(index=False):
            problems = ohlc_problems(
                float(record.Open),
                float(record.High),
                float(record.Low),
                float(record.Close),
                float(record.Volume),
            )
            bar_time = pd.Timestamp(record.Date).isoformat()
            if problems:
                rejected += 1
                for name in problems:
                    self._quality(symbol, timeframe, bar_time, name, "error", name, provider_id, now)
                continue
            flags = []
            if pd.Timestamp(record.Date) in jumps:
                flags.append("price_jump")
                self._quality(
                    symbol,
                    timeframe,
                    bar_time,
                    "price_jump",
                    "warning",
                    f"close-to-close move exceeded {jump_threshold:.0%}",
                    provider_id,
                    now,
                )
            rows.append(
                (
                    symbol,
                    timeframe,
                    bar_time,
                    float(record.Open),
                    float(record.High),
                    float(record.Low),
                    float(record.Close),
                    float(record.Volume),
                    provider_id,
                    1 if adjusted else 0,
                    ",".join(flags),
                    now,
                )
            )
            written += 1
        if rows:
            self._conn.executemany(
                """
                INSERT INTO candles (
                    symbol, timeframe, bar_time, open, high, low, close, volume,
                    provider_id, adjusted, quality_flags, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, bar_time) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    provider_id = excluded.provider_id,
                    adjusted = excluded.adjusted,
                    quality_flags = excluded.quality_flags,
                    ingested_at = excluded.ingested_at
                """,
                rows,
            )
        self._conn.commit()
        return {"written": written, "rejected": rejected}

    def read_candles(
        self,
        symbol: str,
        timeframe: str,
        *,
        adjusted: bool = True,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        timeframe = str(timeframe).strip().lower()
        query = [
            """
            SELECT bar_time AS Date, open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume, quality_flags AS QualityFlags
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND adjusted = ?
            """
        ]
        params: list[object] = [symbol, timeframe, 1 if adjusted else 0]
        if start is not None:
            query.append("AND bar_time >= ?")
            params.append(pd.Timestamp(start).isoformat())
        if end is not None:
            query.append("AND bar_time <= ?")
            params.append(pd.Timestamp(end).isoformat())
        query.append("ORDER BY bar_time")
        frame = pd.read_sql_query(" ".join(query), self._conn, params=params)
        if frame.empty:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        frame["Date"] = pd.to_datetime(frame["Date"])
        return frame

    def latest_bar_time(self, symbol: str, timeframe: str, *, adjusted: bool = True) -> pd.Timestamp | None:
        row = self._conn.execute(
            """
            SELECT MAX(bar_time) AS bar_time
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND adjusted = ?
            """,
            (symbol.strip().upper(), str(timeframe).strip().lower(), 1 if adjusted else 0),
        ).fetchone()
        if row is None or row["bar_time"] is None:
            return None
        return pd.Timestamp(row["bar_time"])

    def upsert_quotes(self, quotes: list[Quote]) -> int:
        now = _utc_now()
        payload = []
        for quote in quotes:
            if quote.freshness == "live" and not quote.live_authorized:
                raise ValueError("Refusing to store an unauthorized live quote.")
            payload.append(
                (
                    quote.symbol.strip().upper(),
                    pd.Timestamp(quote.quote_time).isoformat(),
                    float(quote.last_price),
                    quote.bid,
                    quote.ask,
                    quote.volume,
                    quote.provider_id,
                    quote.freshness,
                    now,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO quotes (
                symbol, quote_time, last_price, bid, ask, volume, provider_id, freshness, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, quote_time, provider_id) DO UPDATE SET
                last_price = excluded.last_price,
                bid = excluded.bid,
                ask = excluded.ask,
                volume = excluded.volume,
                freshness = excluded.freshness,
                received_at = excluded.received_at
            """,
            payload,
        )
        self._conn.commit()
        return len(payload)

    def latest_quotes(self, symbols: list[str] | None = None) -> pd.DataFrame:
        if symbols:
            marks = ",".join("?" for _ in symbols)
            query = f"""
                SELECT q.*
                FROM quotes q
                JOIN (
                    SELECT symbol, provider_id, MAX(quote_time) AS quote_time
                    FROM quotes
                    WHERE symbol IN ({marks})
                    GROUP BY symbol, provider_id
                ) latest
                ON q.symbol = latest.symbol
                AND q.provider_id = latest.provider_id
                AND q.quote_time = latest.quote_time
                ORDER BY q.symbol
            """
            return pd.read_sql_query(query, self._conn, params=[s.strip().upper() for s in symbols])
        return pd.read_sql_query(
            """
            SELECT q.*
            FROM quotes q
            JOIN (
                SELECT symbol, provider_id, MAX(quote_time) AS quote_time
                FROM quotes
                GROUP BY symbol, provider_id
            ) latest
            ON q.symbol = latest.symbol
            AND q.provider_id = latest.provider_id
            AND q.quote_time = latest.quote_time
            ORDER BY q.symbol
            """,
            self._conn,
        )

    def record_provider_event(self, provider_id: str, level: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO provider_events (provider_id, level, message, created_at) VALUES (?, ?, ?, ?)",
            (provider_id, level, message, _utc_now()),
        )
        self._conn.commit()

    def provider_events(self, limit: int = 50) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM provider_events ORDER BY id DESC LIMIT ?",
            self._conn,
            params=[int(limit)],
        )

    def quality_events(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM quality_events ORDER BY id DESC LIMIT ?",
            self._conn,
            params=[int(limit)],
        )

    def coverage(self) -> pd.DataFrame:
        return pd.read_sql_query(
            """
            SELECT symbol, timeframe, COUNT(*) AS bars, MIN(bar_time) AS first_bar, MAX(bar_time) AS last_bar,
                   provider_id, adjusted
            FROM candles
            GROUP BY symbol, timeframe, provider_id, adjusted
            ORDER BY symbol, timeframe
            """,
            self._conn,
        )

    def stats(self) -> dict:
        def _count(table: str) -> int:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "path": str(self.path),
            "schema_version": self.migrate(),
            "instruments": _count("instruments"),
            "candles": _count("candles"),
            "quotes": _count("quotes"),
            "quality_events": _count("quality_events"),
            "provider_events": _count("provider_events"),
        }

    def _quality(
        self,
        symbol: str,
        timeframe: str,
        bar_time: str,
        check_name: str,
        severity: str,
        detail: str,
        provider_id: str,
        detected_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO quality_events (
                symbol, timeframe, bar_time, check_name, severity, detail, provider_id, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, timeframe, bar_time, check_name, severity, detail, provider_id, detected_at),
        )
