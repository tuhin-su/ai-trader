"""
services/angel_api.py
=====================
Async wrapper around the Angel One SmartAPI.
Handles auth (TOTP + session), order placement, and market data.

Install: pip install smartapi-python pyotp
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyotp

from config.settings import (
    ANGEL_API_KEY, ANGEL_CLIENT_ID,
    ANGEL_PASSWORD, ANGEL_TOTP_SECRET,
    ENABLE_TRAINING,
)

logger = logging.getLogger(__name__)

# Lazy import — SmartAPI not needed in training-only environments
try:
    from SmartApi import SmartConnect
    _SMARTAPI_AVAILABLE = True
except ImportError:
    _SMARTAPI_AVAILABLE = False
    logger.warning("SmartApi not installed — Angel One API unavailable")


class AngelAPI:
    """Async facade over SmartConnect."""

    def __init__(self) -> None:
        self._obj: Optional[object] = None
        self._session_token: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._connected = False

    # ── Connection ───────────────────────────────────────────

    async def connect(self) -> None:
        if ENABLE_TRAINING:
            logger.info("AngelAPI: training mode — skipping real connection")
            return
        if not _SMARTAPI_AVAILABLE:
            raise ImportError("Install smartapi-python: pip install smartapi-python pyotp")
        if not ANGEL_API_KEY:
            raise ValueError("ANGEL_API_KEY not set in .env")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_connect)

    def _sync_connect(self) -> None:
        self._obj = SmartConnect(api_key=ANGEL_API_KEY)
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        data = self._obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if data["status"] is False:
            raise ConnectionError(f"Angel One login failed: {data['message']}")
        self._session_token = data["data"]["jwtToken"]
        self._feed_token    = data["data"]["feedToken"]
        self._connected = True
        logger.info("Angel One session established")

    # ── Market Data ──────────────────────────────────────────

    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[dict]:
        """Live quote for a single symbol."""
        if not self._connected:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_quote, symbol, exchange)

    def _sync_quote(self, symbol: str, exchange: str) -> dict:
        try:
            resp = self._obj.ltpData(exchange, symbol, self._get_token(symbol))
            return resp.get("data", {})
        except Exception as exc:
            logger.error(f"Quote error for {symbol}: {exc}")
            return {}

    async def get_historical(
        self,
        symbol: str,
        interval: str = "ONE_DAY",
        days: int = 365,
        exchange: str = "NSE",
    ) -> Optional[pd.DataFrame]:
        """OHLCV DataFrame for historical data."""
        if not self._connected:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_historical, symbol, interval, days, exchange
        )

    def _sync_historical(
        self, symbol: str, interval: str, days: int, exchange: str
    ) -> pd.DataFrame:
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            params = {
                "exchange":    exchange,
                "symboltoken": self._get_token(symbol),
                "interval":    interval,
                "fromdate":    from_date.strftime("%Y-%m-%d %H:%M"),
                "todate":      to_date.strftime("%Y-%m-%d %H:%M"),
            }
            resp = self._obj.getCandleData(params)
            candles = resp.get("data", [])
            df = pd.DataFrame(
                candles,
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as exc:
            logger.error(f"Historical data error for {symbol}: {exc}")
            return pd.DataFrame()

    # ── Order Placement ──────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        action: str,         # "BUY" | "SELL"
        quantity: int,
        order_type: str = "MARKET",
        exchange: str = "NSE",
        product: str = "INTRADAY",
    ) -> Optional[str]:
        """
        Place a real order.  Returns order_id if successful.
        ALWAYS guarded against training mode in ExecutionEngine.
        """
        assert not ENABLE_TRAINING, "BUG: place_order called in training mode!"
        if not self._connected:
            raise ConnectionError("Not connected to Angel One")

        order_params = {
            "variety":          "NORMAL",
            "tradingsymbol":    symbol,
            "symboltoken":      self._get_token(symbol),
            "transactiontype":  action,
            "exchange":         exchange,
            "ordertype":        order_type,
            "producttype":      product,
            "duration":         "DAY",
            "quantity":         quantity,
        }
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_place_order, order_params)

    def _sync_place_order(self, params: dict) -> Optional[str]:
        try:
            resp = self._obj.placeOrder(params)
            if resp.get("status"):
                order_id = resp["data"]["orderid"]
                logger.info(f"Order placed: {order_id}")
                return order_id
            else:
                logger.error(f"Order failed: {resp.get('message')}")
                return None
        except Exception as exc:
            logger.error(f"place_order exception: {exc}")
            return None

    # ── Helpers ──────────────────────────────────────────────

    def _get_token(self, symbol: str) -> str:
        """
        Resolve symbol → Angel One token number.
        In production, use a pre-loaded token CSV from Angel One's instrument list.
        """
        # Minimal hardcoded fallback for common Nifty 50 symbols
        _TOKENS = {
            "RELIANCE":  "2885",
            "INFY":      "1594",
            "TCS":       "11536",
            "HDFCBANK":  "1333",
            "ICICIBANK": "4963",
        }
        return _TOKENS.get(symbol.upper(), "0")
