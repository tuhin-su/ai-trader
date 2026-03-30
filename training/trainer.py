"""
training/trainer.py
===================
Orchestrates the full training pipeline:
  1. Build / load dataset
  2. Create TradingEnv
  3. Train (or continue training) the RL agent
  4. Evaluate on a hold-out slice
  5. Save model
"""
from __future__ import annotations

import logging
import numpy as np
from datetime import datetime

from config.settings import TRAIN_TIMESTEPS
from env.trading_env import TradingEnv
from models.model_manager import ModelManager
from training.dataset_builder import DatasetBuilder
from utils.console import log_info, log_warn

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, db=None) -> None:
        self.db = db
        self.model_mgr = ModelManager()

    async def run(self) -> None:
        log_info("=== Training pipeline started ===")

        # 1. Fetch data
        builder = DatasetBuilder()
        train_data, val_data = await builder.build()
        log_info(
            f"Dataset ready: train={len(train_data)} rows, val={len(val_data)} rows"
        )

        # 2. Build environment
        train_env = TradingEnv(data=train_data)

        # 3. Load existing model or train fresh
        agent = self.model_mgr.load_or_train(env=train_env)

        # 4. Evaluate on validation set
        val_metrics = self._evaluate(agent, val_data)
        log_info(
            f"Validation | "
            f"Total return: {val_metrics['total_return']:+.2%} | "
            f"Max drawdown: {val_metrics['max_drawdown']:.2%} | "
            f"Trades: {val_metrics['trade_count']}"
        )

        if self.db:
            await self.db.log_training(
                timestamp=datetime.utcnow().isoformat(),
                timesteps=TRAIN_TIMESTEPS,
                **val_metrics,
            )

        log_info("=== Training pipeline complete ===")

    # ── Private ──────────────────────────────────────────────

    def _evaluate(self, agent, data: np.ndarray) -> dict:
        """Run one deterministic episode on the validation set."""
        env = TradingEnv(data=data)
        obs, _ = env.reset()
        done = False

        portfolio_history = [env.initial_capital]

        while not done:
            action, _ = agent.predict(obs)
            obs, _, done, _, info = env.step(action)
            portfolio_history.append(info["portfolio_value"])

        peak = 0.0
        max_dd = 0.0
        for pv in portfolio_history:
            peak = max(peak, pv)
            dd = (peak - pv) / (peak + 1e-8)
            max_dd = max(max_dd, dd)

        final = portfolio_history[-1]
        total_return = (final - env.initial_capital) / env.initial_capital

        return {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "final_portfolio": final,
            "trade_count": info["trade_count"],
        }
