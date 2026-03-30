"""
services/scanner.py
====================
Async market scanner — fetches quotes for the watchlist and filters by
momentum / liquidity criteria before passing candidates to the AI.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from config.settings import SCAN_LIMIT
from services.stock_universe import StockUniverse
from utils.indicators import Indicators

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    symbol: str
    price: float
    features: np.ndarray   # ready for RLAgent.predict()


class Scanner:
    def __init__(self, api) -> None:
        self.api = api
        self.universe = StockUniverse()

    async def scan(self) -> List[Candidate]:
        symbols = self.universe.get_watchlist()[:SCAN_LIMIT]
        tasks = [self._analyse(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = [r for r in results if isinstance(r, Candidate)]
        logger.info(f"Scanner: {len(candidates)}/{len(symbols)} passed filters")
        return candidates

    async def _analyse(self, symbol: str):
        try:
            hist = await self.api.get_historical(symbol, interval="ONE_HOUR", days=30)
            if hist is None or len(hist) < 50:
                return None
            df = Indicators.add_all(hist)
            df = df.dropna()
            if not self._passes_filter(df):
                return None
            features = Indicators.feature_vector(df, window=30)
            price = float(df["close"].iloc[-1])
            return Candidate(symbol=symbol, price=price, features=features)
        except Exception as exc:
            logger.debug(f"Scanner error for {symbol}: {exc}")
            return None

    @staticmethod
    def _passes_filter(df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        # Basic momentum + liquidity gate
        vol_ok   = df["volume"].tail(5).mean() > 100_000
        rsi_ok   = 30 < last.get("rsi", 50) < 70
        trend_ok = last.get(f"ema_9", last["close"]) > last.get(f"ema_21", 0)
        return bool(vol_ok and rsi_ok and trend_ok)
