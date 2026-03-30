"""
models/model_manager.py
=======================
Manages the lifecycle of the RL model:
  • Load if exists (live mode / continued training)
  • Train fresh if none found (first-run training mode)
  • Continual-learning retrain from accumulated trade data
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import MODEL_PATH, RETRAIN_INTERVAL_TRADES
from models.rl_agent import RLAgent

logger = logging.getLogger(__name__)


class ModelManager:
    """Central point for model persistence and lifecycle."""

    def __init__(self) -> None:
        self._agent: Optional[RLAgent] = None

    # ── Public API ───────────────────────────────────────────

    def load_or_train(self, env) -> RLAgent:
        """Load existing model or train a fresh one on *env*."""
        agent = RLAgent(env=env)
        if self._model_exists():
            logger.info("Found existing model — loading")
            agent.load()
        else:
            logger.info("No model found — training from scratch")
            agent.build()
            agent.train()
            agent.save()
        self._agent = agent
        return agent

    def load_or_raise(self) -> RLAgent:
        """Load existing model; raise if none found (live mode guard)."""
        if not self._model_exists():
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. "
                "Run in TRAINING mode first to build a model."
            )
        agent = RLAgent()
        agent.load()
        self._agent = agent
        return agent

    def maybe_retrain(self, new_data: np.ndarray, trade_count: int) -> None:
        """
        Trigger a continual-learning pass when enough new trades accumulated.
        Called from the live pipeline after each batch of live trades.
        """
        if trade_count % RETRAIN_INTERVAL_TRADES != 0:
            return
        if self._agent is None:
            logger.warning("No agent to retrain")
            return

        logger.info(
            f"Continual learning triggered at {trade_count} trades — "
            f"retraining on {len(new_data)} new rows"
        )
        from env.trading_env import TradingEnv
        new_env = TradingEnv(data=new_data)
        self._agent.continue_training(new_env=new_env, timesteps=50_000)
        self._agent.save()
        logger.info("Continual learning complete — model updated")

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _model_exists() -> bool:
        # SB3 saves as "<path>.zip"
        return Path(MODEL_PATH + ".zip").exists()

    @property
    def agent(self) -> Optional[RLAgent]:
        return self._agent
