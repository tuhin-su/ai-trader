"""
services/stock_universe.py
===========================
Manages the tradeable watchlist.  Loads from data/stocks.json if present,
otherwise falls back to a hardcoded Nifty 50 subset.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from config.settings import BASE_DIR

logger = logging.getLogger(__name__)

_DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "KOTAKBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "TITAN", "LT",
    "AXISBANK", "HCLTECH", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
]

_STOCKS_FILE = BASE_DIR / "data" / "stocks.json"


class StockUniverse:
    def __init__(self) -> None:
        self._watchlist: List[str] = self._load()

    def get_watchlist(self) -> List[str]:
        return list(self._watchlist)

    def add(self, symbol: str) -> None:
        if symbol not in self._watchlist:
            self._watchlist.append(symbol)
            self._save()

    def remove(self, symbol: str) -> None:
        self._watchlist = [s for s in self._watchlist if s != symbol]
        self._save()

    def _load(self) -> List[str]:
        if _STOCKS_FILE.exists():
            try:
                data = json.loads(_STOCKS_FILE.read_text())
                return data.get("watchlist", _DEFAULT_WATCHLIST)
            except Exception as exc:
                logger.warning(f"stocks.json load error: {exc}")
        return list(_DEFAULT_WATCHLIST)

    def _save(self) -> None:
        _STOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STOCKS_FILE.write_text(json.dumps({"watchlist": self._watchlist}, indent=2))
