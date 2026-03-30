"""
training/continuous_learner.py
================================
Background coroutine that runs alongside the live pipeline.
Every N trades it fetches the latest reward history from the DB,
reconstructs a mini feature matrix, and triggers a continual-learning
pass on the RL agent so the model adapts to recent market conditions.

Usage (inside run_live_pipeline):
    learner = ContinuousLearner(model_mgr=model_mgr, db=db)
    asyncio.create_task(learner.run())
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from config.settings import RETRAIN_INTERVAL_TRADES, ENABLE_TRAINING

logger = logging.getLogger(__name__)

_CHECK_INTERVAL = 300   # seconds between checks (5 minutes)


class ContinuousLearner:
    def __init__(self, model_mgr, db) -> None:
        self.model_mgr  = model_mgr
        self.db         = db
        self._last_count = 0

    async def run(self) -> None:
        """Long-running background task — call via asyncio.create_task()."""
        logger.info("ContinuousLearner started")
        while True:
            try:
                await asyncio.sleep(_CHECK_INTERVAL)
                await self._maybe_retrain()
            except asyncio.CancelledError:
                logger.info("ContinuousLearner cancelled")
                break
            except Exception as exc:
                logger.error(f"ContinuousLearner error: {exc}")

    async def _maybe_retrain(self) -> None:
        # Count total trades since last retrain trigger
        rows = await self.db.get_recent_rewards(limit=10_000)
        count = len(rows)

        new_trades = count - self._last_count
        if new_trades < RETRAIN_INTERVAL_TRADES:
            logger.debug(f"ContinuousLearner: {new_trades} new trades, need {RETRAIN_INTERVAL_TRADES}")
            return

        logger.info(f"ContinuousLearner: {new_trades} new trades → triggering retrain")

        # Build a tiny synthetic feature array from reward observations
        # In production: reconstruct proper OHLCV + indicators from stored trades
        if len(rows) >= 50:
            arr = self._rewards_to_features(rows)
            self.model_mgr.maybe_retrain(new_data=arr, trade_count=count)
            self._last_count = count

    @staticmethod
    def _rewards_to_features(rows: list) -> np.ndarray:
        """
        Convert raw reward rows (symbol, action, pnl) → feature array.
        This is a placeholder — replace with real OHLCV + indicator
        reconstruction in production.
        """
        pnls = np.array([r[2] for r in rows], dtype=np.float32)
        # Simulate a 9-column feature matrix by repeating and jittering pnl
        n = len(pnls)
        base = pnls.reshape(-1, 1) / (np.abs(pnls).max() + 1e-9)
        noise = np.random.normal(0, 0.05, (n, 8)).astype(np.float32)
        return np.hstack([base, noise])
