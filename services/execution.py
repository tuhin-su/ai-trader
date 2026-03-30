"""
services/execution.py
======================
Execution engine — the ONLY place where real orders can be placed.

CRITICAL SAFETY CONTRACT
  • This module checks ENABLE_TRAINING before EVERY real order call.
  • If ENABLE_TRAINING is True, only simulate_trade() is ever called.
  • Real orders go through api.place_order() ONLY when ENABLE_LIVE_TRADING=True.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from config.settings import (
    ENABLE_TRAINING, ENABLE_LIVE_TRADING,
    CAPITAL, MAX_RISK_PER_TRADE, STOP_LOSS_PCT, TARGET_PCT
)

logger = logging.getLogger(__name__)

# Actions
HOLD = 0
BUY  = 1
SELL = 2

_ACTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}


class ExecutionEngine:
    def __init__(self, api=None, db=None, telegram=None) -> None:
        self.api      = api
        self.db       = db
        self.telegram = telegram
        self._sim_positions: dict = {}   # symbol → entry_price (simulation)

    # ── Public API ───────────────────────────────────────────

    async def execute(
        self,
        symbol: str,
        action: int,
        confidence: float,
        price: Optional[float] = None,
    ) -> None:
        """Route to simulation or live based on mode flag."""
        # ── SAFETY GATE ─────────────────────────────────────
        if ENABLE_TRAINING and ENABLE_LIVE_TRADING:
            raise RuntimeError("CRITICAL: both training and live modes are True!")

        if ENABLE_TRAINING:
            await self._simulate_trade(symbol, action, confidence, price)
        elif ENABLE_LIVE_TRADING:
            await self._live_trade(symbol, action, confidence, price)
        else:
            raise RuntimeError("Neither training nor live mode is active.")

    # ── Simulation ───────────────────────────────────────────

    async def _simulate_trade(
        self,
        symbol: str,
        action: int,
        confidence: float,
        price: Optional[float],
    ) -> None:
        entry = price or self._sim_positions.get(symbol, 100.0)

        if action == BUY:
            self._sim_positions[symbol] = entry
            logger.info(f"[SIM] BUY  {symbol} @ {entry:.2f} conf={confidence:.2%}")

        elif action == SELL and symbol in self._sim_positions:
            entry_p = self._sim_positions.pop(symbol)
            pnl     = (entry - entry_p) / entry_p
            logger.info(
                f"[SIM] SELL {symbol} @ {entry:.2f} "
                f"entry={entry_p:.2f} PnL={pnl:+.2%} conf={confidence:.2%}"
            )
            if self.db:
                await self.db.save_trade(
                    symbol=symbol, action="SELL", price=entry,
                    entry_price=entry_p, pnl=pnl,
                    mode="simulation", timestamp=datetime.utcnow().isoformat(),
                )

        if self.db:
            await self.db.update_reward(symbol=symbol, action=action, pnl=0.0)

    # ── Live ─────────────────────────────────────────────────

    async def _live_trade(
        self,
        symbol: str,
        action: int,
        confidence: float,
        price: Optional[float],
    ) -> None:
        assert not ENABLE_TRAINING, "BUG: live trade attempted in training mode"

        action_str = _ACTION_NAMES[action]
        qty = self._calc_quantity(price or 1.0)
        if qty <= 0:
            logger.warning(f"[LIVE] Skipping {symbol} — qty 0")
            return

        logger.info(f"[LIVE] Placing {action_str} {qty}×{symbol} conf={confidence:.2%}")

        order_id = await self.api.place_order(
            symbol=symbol,
            action=action_str,
            quantity=qty,
        )

        if order_id:
            sl_price  = (price or 1.0) * (1 - STOP_LOSS_PCT)
            tgt_price = (price or 1.0) * (1 + TARGET_PCT)
            msg = (
                f"📈 {action_str} {symbol}\n"
                f"Qty: {qty}  |  Price: {price:.2f}\n"
                f"SL: {sl_price:.2f}  |  Target: {tgt_price:.2f}\n"
                f"Confidence: {confidence:.0%}\n"
                f"Order ID: {order_id}"
            )
            if self.telegram:
                await self.telegram.send_message(msg)
            if self.db:
                await self.db.save_trade(
                    symbol=symbol, action=action_str, price=price or 0.0,
                    entry_price=price or 0.0, pnl=0.0,
                    mode="live", order_id=order_id,
                    timestamp=datetime.utcnow().isoformat(),
                )
        else:
            logger.error(f"[LIVE] Order placement failed for {symbol}")

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _calc_quantity(price: float) -> int:
        risk_amount = CAPITAL * MAX_RISK_PER_TRADE
        qty = int(risk_amount / (price * STOP_LOSS_PCT + 1e-9))
        return max(1, min(qty, 500))   # floor 1, ceiling 500 shares
