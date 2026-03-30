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
    BASE_DIR, TRAIN_SYMBOLS, TRAIN_DAYS, DATA_SOURCE
)
from utils.indicators import Indicators

logger = logging.getLogger(__name__)

CSV_DIR = BASE_DIR / "data" / "csv"
TRAIN_RATIO = 0.8


class DatasetBuilder:
    """Builds train/val numpy arrays from historical OHLCV data."""

    async def build(self) -> Tuple[np.ndarray, np.ndarray]:
        all_frames = []
        
        for symbol in TRAIN_SYMBOLS:
            df = await self._load_data(symbol)
            if df is not None and not df.empty:
                # Add indicators PER SYMBOL to avoid leakage across boundaries
                df = Indicators.add_all(df)
                df = df.dropna().reset_index(drop=True)
                all_frames.append(df)
        
        if not all_frames:
            logger.error("No data loaded for any symbol. Falling back to synthetic.")
            combined = self._synthetic_data()
            combined = Indicators.add_all(combined).dropna()
        else:
            combined = pd.concat(all_frames).reset_index(drop=True)

        feature_cols = [
            "close_norm", "ema_short_norm", "ema_long_norm",
            "rsi_norm", "volume_norm", "volatility_norm",
            "atr_norm", "bb_upper_norm", "bb_lower_norm",
        ]
        # Ensure all feature columns exist (fallback to zeros if missing)
        for col in feature_cols:
            if col not in combined.columns:
                combined[col] = 0.0

        arr = combined[feature_cols].values.astype(np.float32)
        # Safety: Ensure no NaNs or Infs leak into training
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        split = int(len(arr) * TRAIN_RATIO)
        logger.info(f"Dataset ready: {len(arr)} total rows ({split} train, {len(arr)-split} val)")
        return arr[:split], arr[split:]

    # ── Private ──────────────────────────────────────────────

    async def _load_data(self, symbol: str) -> pd.DataFrame:
        """Load data based on DATA_SOURCE setting with fallback."""
        source = DATA_SOURCE.upper()
        
        if source == "ANGEL_API":
            return await self._load_from_api(symbol)
        elif source == "LOCAL_CSV":
            return self._load_from_csv(symbol)
        else:
            return self._synthetic_data()

    async def _load_from_api(self, symbol: str) -> pd.DataFrame:
        try:
            from services.angel_api import AngelAPI
            api = AngelAPI()
            await api.connect()
            df = await api.get_historical(symbol, interval="ONE_DAY", days=TRAIN_DAYS)
            if df is not None and len(df) > 100:
                logger.info(f"Loaded {len(df)} rows for {symbol} from Angel One API")
                return df
        except Exception as exc:
            logger.warning(f"API fetch failed for {symbol}: {exc} — falling back to CSV")
        return self._load_from_csv(symbol)

    def _load_from_csv(self, symbol: str) -> pd.DataFrame:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        # Try matching symbol-specific CSV first
        pattern = f"*{symbol}*.csv"
        csv_files = sorted(CSV_DIR.glob(pattern))
        
        if not csv_files and symbol == TRAIN_SYMBOLS[0]:
            # Fallback to all CSVs only if it's the first symbol and no specific match found
            csv_files = sorted(CSV_DIR.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSVs found for {symbol} — generating synthetic data")
            return self._synthetic_data()

        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, parse_dates=["date"])
                frames.append(df)
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
        
        if not frames: return self._synthetic_data()
        
        combined = pd.concat(frames).sort_values("date").reset_index(drop=True)
        logger.info(f"Loaded {len(combined)} rows for {symbol} from CSV")
        return combined

    @staticmethod
    def _synthetic_data(n: int = 1000) -> pd.DataFrame:
        """Brownian-motion price series for unit testing when no data available."""
        # Using a semi-random seed based on time to vary synthetic data slightly per call
        import time
        np.random.seed(int(time.time()) % 10000)
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
