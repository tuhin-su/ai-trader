"""
utils/indicators.py
===================
Vectorised technical indicators built on Pandas / NumPy.
All functions accept a OHLCV DataFrame and return a new DataFrame
with additional columns appended.

Convention: columns ending in _norm are min-max normalised [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    EMA_SHORT, EMA_LONG, RSI_PERIOD,
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD
)


class Indicators:
    """Static factory — call Indicators.add_all(df)."""

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = Indicators.add_ema(df)
        df = Indicators.add_rsi(df)
        df = Indicators.add_atr(df)
        df = Indicators.add_bollinger(df)
        df = Indicators.add_vwap(df)
        df = Indicators.add_volatility(df)
        df = Indicators.add_normalised(df)
        return df

    # ── EMA ──────────────────────────────────────────────────

    @staticmethod
    def add_ema(df: pd.DataFrame) -> pd.DataFrame:
        df[f"ema_{EMA_SHORT}"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
        df[f"ema_{EMA_LONG}"]  = df["close"].ewm(span=EMA_LONG,  adjust=False).mean()
        return df

    # ── RSI ──────────────────────────────────────────────────

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    # ── ATR ──────────────────────────────────────────────────

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=period, adjust=False).mean()
        return df

    # ── Bollinger Bands ──────────────────────────────────────

    @staticmethod
    def add_bollinger(
        df: pd.DataFrame,
        period: int = BOLLINGER_PERIOD,
        std_mult: float = BOLLINGER_STD,
    ) -> pd.DataFrame:
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["bb_upper"] = ma + std_mult * std
        df["bb_lower"] = ma - std_mult * std
        df["bb_mid"]   = ma
        return df

    # ── VWAP ─────────────────────────────────────────────────

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].replace(0, 1)
        df["vwap"] = (typical * vol).cumsum() / vol.cumsum()
        return df

    # ── Volatility ───────────────────────────────────────────

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = log_ret.rolling(window).std() * np.sqrt(252)
        return df

    # ── Normalisation ────────────────────────────────────────

    @staticmethod
    def add_normalised(df: pd.DataFrame) -> pd.DataFrame:
        def minmax(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn + 1e-9)

        col_map = {
            "close":          "close_norm",
            f"ema_{EMA_SHORT}": "ema_short_norm",
            f"ema_{EMA_LONG}":  "ema_long_norm",
            "rsi":            "rsi_norm",
            "volume":         "volume_norm",
            "volatility":     "volatility_norm",
            "atr":            "atr_norm",
            "bb_upper":       "bb_upper_norm",
            "bb_lower":       "bb_lower_norm",
        }
        for src, dst in col_map.items():
            if src in df.columns:
                df[dst] = minmax(df[src])
        return df

    # ── Convenience: extract feature vector for live inference ──

    @staticmethod
    def feature_vector(df: pd.DataFrame, window: int = 30) -> "np.ndarray":
        """
        Returns the last *window* rows of normalised features as a 1-D array
        suitable for passing to RLAgent.predict().
        """
        import numpy as np
        norm_cols = [
            "close_norm", "ema_short_norm", "ema_long_norm",
            "rsi_norm", "volume_norm", "volatility_norm",
            "atr_norm", "bb_upper_norm", "bb_lower_norm",
        ]
        available = [c for c in norm_cols if c in df.columns]
        arr = df[available].tail(window).values.astype(np.float32)
        # Pad if fewer rows than window
        if len(arr) < window:
            pad = np.zeros((window - len(arr), len(available)), dtype=np.float32)
            arr = np.vstack([pad, arr])
        # Append position_flag=0, pnl_norm=0 (updated by caller if needed)
        return np.concatenate([arr.flatten(), [0.0, 0.0]])
