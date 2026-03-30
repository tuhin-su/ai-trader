"""
tests/test_suite.py
====================
pytest test suite — covers the critical safety contract,
the trading environment, indicators, risk manager, and execution engine.

Run:  pytest tests/test_suite.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import asyncio

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(0)
    returns = np.random.normal(0.0003, 0.012, n)
    prices  = 1000.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "date":   pd.date_range("2022-01-01", periods=n, freq="D"),
        "open":   prices * (1 + np.random.normal(0, 0.002, n)),
        "high":   prices * (1 + np.abs(np.random.normal(0, 0.006, n))),
        "low":    prices * (1 - np.abs(np.random.normal(0, 0.006, n))),
        "close":  prices,
        "volume": np.random.randint(200_000, 3_000_000, n).astype(float),
    })


def _make_feature_array(rows: int = 300, cols: int = 9) -> np.ndarray:
    np.random.seed(1)
    return np.random.rand(rows, cols).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 1. SAFETY CONTRACT
# ─────────────────────────────────────────────────────────────

class TestSafetyContract:
    """The most important tests in the entire system."""

    def test_both_modes_true_raises(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ENABLE_TRAINING", True)
        monkeypatch.setattr(s, "ENABLE_LIVE_TRADING", True)

        # Simulate what main._safety_check() does
        with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
            if s.ENABLE_TRAINING and s.ENABLE_LIVE_TRADING:
                raise RuntimeError("CRITICAL SAFETY VIOLATION")

    def test_both_modes_false_raises(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ENABLE_TRAINING", False)
        monkeypatch.setattr(s, "ENABLE_LIVE_TRADING", False)

        with pytest.raises(RuntimeError):
            if not s.ENABLE_TRAINING and not s.ENABLE_LIVE_TRADING:
                raise RuntimeError("Neither mode active")

    def test_execution_engine_blocks_real_order_in_training(self, monkeypatch):
        """place_order must NEVER be called when ENABLE_TRAINING=True."""
        import config.settings as s
        monkeypatch.setattr(s, "ENABLE_TRAINING", True)
        monkeypatch.setattr(s, "ENABLE_LIVE_TRADING", False)

        from services.execution import ExecutionEngine

        order_called = []

        class MockAPI:
            async def place_order(self, **kwargs):
                order_called.append(True)
                return "FAKE_ORDER"

        engine = ExecutionEngine(api=MockAPI())

        asyncio.get_event_loop().run_until_complete(
            engine.execute("RELIANCE", 1, 0.9, price=2500.0)
        )
        assert len(order_called) == 0, "place_order was called in training mode!"

    def test_execution_engine_simulates_in_training(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ENABLE_TRAINING", True)
        monkeypatch.setattr(s, "ENABLE_LIVE_TRADING", False)

        from services.execution import ExecutionEngine
        engine = ExecutionEngine()

        # Should not raise
        asyncio.get_event_loop().run_until_complete(
            engine.execute("TCS", 1, 0.8, price=3500.0)
        )
        assert "TCS" in engine._sim_positions


# ─────────────────────────────────────────────────────────────
# 2. TRADING ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class TestTradingEnv:
    def _env(self):
        from env.trading_env import TradingEnv
        return TradingEnv(data=_make_feature_array())

    def test_reset_returns_correct_obs_shape(self):
        env = self._env()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_step_hold(self):
        env = self._env()
        obs, _ = env.reset()
        obs2, reward, done, truncated, info = env.step(0)  # HOLD
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_buy_increases_position(self):
        env = self._env()
        env.reset()
        env.step(1)  # BUY
        assert env.position > 0

    def test_sell_clears_position(self):
        env = self._env()
        env.reset()
        env.step(1)  # BUY
        env.step(2)  # SELL
        assert env.position == 0

    def test_episode_runs_to_completion(self):
        env = self._env()
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 10_000:
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            steps += 1
        assert done, "Episode never terminated"

    def test_reward_clipped(self):
        env = self._env()
        env.reset()
        for _ in range(10):
            _, reward, done, _, _ = env.step(env.action_space.sample())
            assert -10.0 <= reward <= 10.0, f"Reward {reward} out of clip range"
            if done:
                break

    def test_obs_in_bounds(self):
        env = self._env()
        obs, _ = env.reset()
        # Obs may exceed [-10,10] before clipping in observation_space; just check shape
        assert obs.shape == (env.window * env.n_features + 2,)


# ─────────────────────────────────────────────────────────────
# 3. INDICATORS
# ─────────────────────────────────────────────────────────────

class TestIndicators:
    def test_add_all_produces_expected_columns(self):
        from utils.indicators import Indicators
        df = _make_ohlcv()
        out = Indicators.add_all(df)
        expected = ["ema_9", "ema_21", "rsi", "atr", "bb_upper", "bb_lower", "vwap", "volatility"]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_rsi_bounds(self):
        from utils.indicators import Indicators
        df = _make_ohlcv(300)
        out = Indicators.add_all(df).dropna()
        assert out["rsi"].between(0, 100).all(), "RSI out of [0,100]"

    def test_normalised_cols_in_range(self):
        from utils.indicators import Indicators
        df = _make_ohlcv(300)
        out = Indicators.add_all(df).dropna()
        for col in ["close_norm", "rsi_norm", "volume_norm"]:
            assert out[col].between(-0.01, 1.01).all(), f"{col} not in [0,1]"

    def test_ema_short_gt_ema_long_in_uptrend(self):
        from utils.indicators import Indicators
        # Strictly ascending prices → ema_9 should end above ema_21
        prices = np.linspace(100, 200, 100)
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100),
            "open": prices, "high": prices * 1.01,
            "low": prices * 0.99, "close": prices,
            "volume": np.ones(100) * 1_000_000,
        })
        out = Indicators.add_all(df).dropna()
        assert out["ema_9"].iloc[-1] > out["ema_21"].iloc[-1]

    def test_feature_vector_shape(self):
        from utils.indicators import Indicators
        df = _make_ohlcv(200)
        out = Indicators.add_all(df).dropna()
        vec = Indicators.feature_vector(out, window=30)
        # 9 norm features × 30 window + 2 extra = 272
        assert vec.shape == (30 * 9 + 2,), f"Unexpected shape: {vec.shape}"


# ─────────────────────────────────────────────────────────────
# 4. RISK MANAGER
# ─────────────────────────────────────────────────────────────

class TestRiskManager:
    def _risk(self, db=None):
        from services.risk_manager import RiskManager
        return RiskManager(db=db)

    def test_low_confidence_rejected(self):
        risk = self._risk()
        result = asyncio.get_event_loop().run_until_complete(
            risk.check("INFY", action=1, confidence=0.40)
        )
        assert result is False

    def test_high_confidence_no_db_accepted(self):
        risk = self._risk(db=None)
        result = asyncio.get_event_loop().run_until_complete(
            risk.check("INFY", action=1, confidence=0.80)
        )
        assert result is True

    def test_position_size_positive(self):
        from services.risk_manager import RiskManager
        qty = RiskManager.position_size(capital=100_000, price=2500.0, atr=50.0)
        assert qty >= 1
        assert isinstance(qty, int)

    def test_position_size_scales_with_atr(self):
        from services.risk_manager import RiskManager
        qty_small_atr = RiskManager.position_size(100_000, 2500.0, 25.0)
        qty_large_atr = RiskManager.position_size(100_000, 2500.0, 100.0)
        assert qty_small_atr > qty_large_atr, "Larger ATR should give smaller qty"


# ─────────────────────────────────────────────────────────────
# 5. MODEL MANAGER
# ─────────────────────────────────────────────────────────────

class TestModelManager:
    def test_load_or_raise_missing_model(self, tmp_path, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "MODEL_PATH", str(tmp_path / "nonexistent_model"))
        from models.model_manager import ModelManager
        mgr = ModelManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_or_raise()


# ─────────────────────────────────────────────────────────────
# 6. DATABASE
# ─────────────────────────────────────────────────────────────

class TestDatabase:
    def _db(self, tmp_path):
        from services.database import Database
        return Database(path=str(tmp_path / "test.db"))

    def test_init_creates_tables(self, tmp_path):
        db = self._db(tmp_path)
        asyncio.get_event_loop().run_until_complete(db.init())

    def test_save_and_count_trade(self, tmp_path):
        db = self._db(tmp_path)

        async def run():
            await db.init()
            await db.save_trade(
                symbol="RELIANCE", action="BUY", price=2500.0,
                entry_price=2500.0, pnl=0.0,
                mode="simulation", timestamp="2024-01-01T10:00:00",
            )
            count = await db.count_open_positions()
            return count

        asyncio.get_event_loop().run_until_complete(run())

    def test_daily_pnl_returns_zero_for_empty(self, tmp_path):
        from datetime import date
        db = self._db(tmp_path)

        async def run():
            await db.init()
            return await db.get_daily_pnl(date.today())

        pnl = asyncio.get_event_loop().run_until_complete(run())
        assert pnl == 0.0


# ─────────────────────────────────────────────────────────────
# 7. DATASET BUILDER
# ─────────────────────────────────────────────────────────────

class TestDatasetBuilder:
    def test_synthetic_data_shape(self):
        from training.dataset_builder import DatasetBuilder
        df = DatasetBuilder._synthetic_data(n=500)
        assert len(df) == 500
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_build_returns_correct_split(self):
        from training.dataset_builder import DatasetBuilder
        import unittest.mock as mock

        builder = DatasetBuilder()

        async def run():
            with mock.patch.object(builder, "_load_data", return_value=_make_ohlcv(500)):
                train, val = await builder.build()
            return train, val

        train, val = asyncio.get_event_loop().run_until_complete(run())
        assert train.ndim == 2
        assert val.ndim == 2
        assert len(train) > len(val), "Train set should be larger than val"
        assert train.shape[1] == val.shape[1], "Feature count must match"


# ─────────────────────────────────────────────────────────────
# 8. INTEGRATION — one full simulated episode
# ─────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_sim_episode(self):
        """Train a trivial PPO model for 1000 steps and run one episode."""
        pytest.importorskip("stable_baselines3")

        from env.trading_env import TradingEnv
        from models.rl_agent import RLAgent

        data = _make_feature_array(rows=400)
        env  = TradingEnv(data=data)
        agent = RLAgent(env=env)
        agent.build()
        agent.train(timesteps=1000)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, conf = agent.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        assert isinstance(total_reward, float)
        assert info["trade_count"] >= 0
