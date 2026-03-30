"""
services/database.py
====================
Async SQLite wrapper using aiosqlite.

Tables
------
  trades          — every executed / simulated trade
  positions       — current open positions
  balance         — capital snapshots
  training_logs   — per-epoch training metrics
  rewards         — RL reward history for continual learning
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import aiosqlite

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    price       REAL    NOT NULL,
    entry_price REAL,
    pnl         REAL    DEFAULT 0,
    mode        TEXT    NOT NULL,   -- 'live' | 'simulation'
    order_id    TEXT,
    timestamp   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL UNIQUE,
    qty         INTEGER NOT NULL,
    entry_price REAL    NOT NULL,
    opened_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS balance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    capital     REAL    NOT NULL,
    equity      REAL    NOT NULL,
    pnl         REAL    NOT NULL,
    snapshot_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS training_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    timesteps       INTEGER NOT NULL,
    total_return    REAL,
    max_drawdown    REAL,
    final_portfolio REAL,
    trade_count     INTEGER
);

CREATE TABLE IF NOT EXISTS rewards (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    action    INTEGER NOT NULL,
    pnl       REAL    NOT NULL,
    logged_at TEXT    DEFAULT (datetime('now'))
);
"""


class Database:
    def __init__(self, path: str = DB_PATH) -> None:
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        import os, pathlib
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info(f"Database ready: {self.path}")

    # ── Trades ───────────────────────────────────────────────

    async def save_trade(self, **kwargs) -> None:
        await self._db.execute(
            """INSERT INTO trades
               (symbol, action, price, entry_price, pnl, mode, order_id, timestamp)
               VALUES (:symbol, :action, :price, :entry_price, :pnl,
                       :mode, :order_id, :timestamp)""",
            {
                "order_id": None,
                "entry_price": None,
                "pnl": 0.0,
                **kwargs,
            },
        )
        await self._db.commit()

    # ── Positions ────────────────────────────────────────────

    async def count_open_positions(self) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM positions"
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def upsert_position(
        self, symbol: str, qty: int, entry_price: float, opened_at: str
    ) -> None:
        await self._db.execute(
            """INSERT INTO positions (symbol, qty, entry_price, opened_at)
               VALUES (?,?,?,?)
               ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty,
               entry_price=excluded.entry_price""",
            (symbol, qty, entry_price, opened_at),
        )
        await self._db.commit()

    async def delete_position(self, symbol: str) -> None:
        await self._db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        await self._db.commit()

    # ── P&L ──────────────────────────────────────────────────

    async def get_daily_pnl(self, day: date) -> float:
        day_str = day.isoformat()
        async with self._db.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE timestamp LIKE ?",
            (f"{day_str}%",),
        ) as cur:
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    # ── Training logs ────────────────────────────────────────

    async def log_training(self, **kwargs) -> None:
        await self._db.execute(
            """INSERT INTO training_logs
               (timestamp, timesteps, total_return, max_drawdown,
                final_portfolio, trade_count)
               VALUES (:timestamp, :timesteps, :total_return, :max_drawdown,
                       :final_portfolio, :trade_count)""",
            kwargs,
        )
        await self._db.commit()

    # ── Rewards (continual learning) ─────────────────────────

    async def update_reward(self, symbol: str, action: int, pnl: float) -> None:
        await self._db.execute(
            "INSERT INTO rewards (symbol, action, pnl) VALUES (?,?,?)",
            (symbol, action, pnl),
        )
        await self._db.commit()

    async def get_recent_rewards(self, limit: int = 5000) -> list:
        async with self._db.execute(
            "SELECT symbol, action, pnl FROM rewards ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            return await cur.fetchall()

    # ── Lifecycle ────────────────────────────────────────────

    async def close(self) -> None:
        if self._db:
            await self._db.close()
