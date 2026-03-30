"""
training/dataset_builder.py
============================
Downloads or loads historical OHLCV data and computes feature columns.
Returns a normalised NumPy array split into (train, val) sets.

Supported sources (configure in settings.py):
  • Angel One historical API
  • Local CSV files in data/csv/
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from config.settings import (
    BASE_DIR, EMA_SHORT, EMA_LONG, RSI_PERIOD,
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD
)
from utils.indicators import Indicators

logger = logging.getLogger(__name__)

CSV_DIR = BASE_DIR / "data" / "csv"
TRAIN_RATIO = 0.8


class DatasetBuilder:
    """Builds train/val numpy arrays from historical OHLCV data."""

    async def build(self) -> Tuple[np.ndarray, np.ndarray]:
        df = await self._load_data()
        df = Indicators.add_all(df)
        df = df.dropna().reset_index(drop=True)

        feature_cols = [
            "close_norm", "ema_short_norm", "ema_long_norm",
            "rsi_norm", "volume_norm", "volatility_norm",
            "atr_norm", "bb_upper_norm", "bb_lower_norm",
        ]
        # Ensure all feature columns exist (fallback to zeros if missing)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        arr = df[feature_cols].values.astype(np.float32)
        # Safety: Ensure no NaNs or Infs leak into training
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        split = int(len(arr) * TRAIN_RATIO)
        return arr[:split], arr[split:]

    # ── Private ──────────────────────────────────────────────

    async def _load_data(self) -> pd.DataFrame:
        """Try Angel One API first; fall back to local CSVs."""
        try:
            from services.angel_api import AngelAPI
            api = AngelAPI()
            await api.connect()
            # Use RELIANCE as default training symbol; extend as needed
            df = await api.get_historical("RELIANCE", interval="ONE_DAY", days=730)
            if df is not None and len(df) > 100:
                logger.info(f"Loaded {len(df)} rows from Angel One API")
                return df
        except Exception as exc:
            logger.warning(f"Angel One historical fetch failed: {exc} — falling back to CSV")

        return self._load_from_csv()

    def _load_from_csv(self) -> pd.DataFrame:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(CSV_DIR.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSVs found — generating synthetic data for testing")
            return self._synthetic_data()

        frames = []
        for f in csv_files:
            df = pd.read_csv(f, parse_dates=["date"])
            frames.append(df)
        combined = pd.concat(frames).sort_values("date").reset_index(drop=True)
        logger.info(f"Loaded {len(combined)} rows from {len(csv_files)} CSV file(s)")
        return combined

    @staticmethod
    def _synthetic_data(n: int = 2000) -> pd.DataFrame:
        """Brownian-motion price series for unit testing when no data available."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, n)
        prices = 1000.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "open":   prices * (1 + np.random.normal(0, 0.003, n)),
            "high":   prices * (1 + np.abs(np.random.normal(0, 0.008, n))),
            "low":    prices * (1 - np.abs(np.random.normal(0, 0.008, n))),
            "close":  prices,
            "volume": np.random.randint(100_000, 5_000_000, n).astype(float),
        })
        return df
