"""
services/risk_manager.py
=========================
Pre-trade risk checks.  Returns True only if a trade is permitted.

Rules enforced:
  1. Max 2% capital per single trade
  2. Max 5% total daily loss
  3. Max 3 simultaneous open positions
  4. Minimum AI confidence threshold
  5. Position-sizing: returns suggested qty based on ATR stop
"""
from __future__ import annotations

import logging
from datetime import date

from config.settings import (
    CAPITAL, MAX_RISK_PER_TRADE, MAX_DAILY_LOSS, MAX_OPEN_TRADES
)

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.55   # Minimum action probability to act


class RiskManager:
    def __init__(self, db=None) -> None:
        self.db = db

    async def check(self, symbol: str, action: int, confidence: float) -> bool:
        """
        Returns True if the proposed trade passes all risk rules.
        action: 1=BUY, 2=SELL
        """
        # 1. Confidence gate
        if confidence < MIN_CONFIDENCE:
            logger.info(f"Risk rejected [{symbol}]: confidence {confidence:.2%} < {MIN_CONFIDENCE:.2%}")
            return False

        if self.db is None:
            return True          # No DB — allow in offline/test mode

        # 2. Max open positions
        open_count = await self.db.count_open_positions()
        if action == 1 and open_count >= MAX_OPEN_TRADES:
            logger.info(f"Risk rejected [{symbol}]: max open positions ({MAX_OPEN_TRADES}) reached")
            return False

        # 3. Daily loss limit
        daily_pnl = await self.db.get_daily_pnl(date.today())
        if daily_pnl <= -MAX_DAILY_LOSS * CAPITAL:
            logger.warning(
                f"Risk rejected [{symbol}]: daily loss {daily_pnl:.0f} exceeds "
                f"{MAX_DAILY_LOSS:.0%} of capital"
            )
            return False

        return True

    @staticmethod
    def position_size(capital: float, price: float, atr: float) -> int:
        """
        ATR-based position sizing.
        Risk 2% of capital on the trade; stop is placed 1 ATR below entry.
        """
        risk_amount = capital * MAX_RISK_PER_TRADE
        stop_distance = atr if atr > 0 else price * 0.01
        qty = int(risk_amount / stop_distance)
        return max(1, qty)
