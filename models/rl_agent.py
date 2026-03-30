"""
models/rl_agent.py
==================
Thin wrapper around Stable Baselines3 algorithms (PPO / SAC / A2C).
Exposes a unified  train() / predict() / save() / load()  interface
so the rest of the system doesn't need to know which algo is in use.
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from config.settings import MODEL_ALGO, MODEL_PATH, TRAIN_TIMESTEPS
from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)

_ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}


class RLAgent:
    """Wraps a Stable Baselines3 model with trading-specific helpers."""

    def __init__(self, env: Optional[TradingEnv] = None) -> None:
        self.env = env
        self.model = None
        self._algo_cls = _ALGO_MAP.get(MODEL_ALGO.upper(), PPO)

    # ── Training ─────────────────────────────────────────────

    def build(self) -> None:
        """Instantiate a fresh model.  Must be called before train()."""
        assert self.env is not None, "Provide an env before calling build()"
        monitored = Monitor(self.env)
        self.model = self._algo_cls(
            policy="MlpPolicy",
            env=monitored,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048 if MODEL_ALGO == "PPO" else None,  # PPO-only
            batch_size=64,
            gamma=0.99,
            tensorboard_log="data/tb_logs/",
            device="auto",
        )
        logger.info(f"Built {MODEL_ALGO} model")

    def train(self, timesteps: int = TRAIN_TIMESTEPS) -> None:
        """Train the model; saves checkpoint every 10 k steps."""
        assert self.model is not None, "Call build() first"
        logger.info(f"Training {MODEL_ALGO} for {timesteps:,} timesteps…")

        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5, min_evals=10, verbose=1
        )
        eval_cb = EvalCallback(
            Monitor(self.env),
            best_model_save_path=MODEL_PATH + "_best/",
            eval_freq=10_000,
            callback_after_eval=stop_cb,
            verbose=1,
        )
        self.model.learn(
            total_timesteps=timesteps,
            callback=eval_cb,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        logger.info("Training complete")

    def continue_training(self, new_env: TradingEnv, timesteps: int = 50_000) -> None:
        """Online / continual learning — pick up from where we left off."""
        assert self.model is not None, "Load a model first"
        self.model.set_env(Monitor(new_env))
        self.model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        logger.info(f"Continued training for {timesteps:,} additional timesteps")

    # ── Inference ────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Returns (action, confidence).
        action: 0=HOLD, 1=BUY, 2=SELL
        confidence: probability of the chosen action (PPO/A2C only; 0.0 for SAC)
        """
        assert self.model is not None, "No model loaded"
        obs = features.astype(np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action[0]) if hasattr(action, "__len__") else int(action)

        # Extract confidence from the policy distribution (PPO / A2C)
        confidence = 0.0
        try:
            import torch
            with torch.no_grad():
                obs_t = self.model.policy.obs_to_tensor(obs)[0]
                dist = self.model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy()[0]
                confidence = float(probs[action])
        except Exception:
            pass  # SAC doesn't have a discrete distribution

        return action, confidence

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        path = path or MODEL_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved → {path}")

    def load(self, path: Optional[str] = None) -> None:
        path = path or MODEL_PATH
        self.model = self._algo_cls.load(path, device="auto")
        logger.info(f"Model loaded ← {path}")
