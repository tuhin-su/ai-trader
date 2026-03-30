"""
services/telegram_service.py
=============================
Async Telegram bot for trade alerts and daily summaries.
"""
from __future__ import annotations

import logging
import aiohttp
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org/bot"


class TelegramService:
    async def send_message(self, text: str) -> None:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.debug("Telegram not configured — message skipped")
            return
        url = f"{_BASE}{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Telegram HTTP {resp.status}")
        except Exception as exc:
            logger.warning(f"Telegram error: {exc}")

    async def send_daily_summary(self, pnl: float, trades: int, portfolio: float) -> None:
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"PnL: {pnl:+,.0f} INR\n"
            f"Trades: {trades}\n"
            f"Portfolio: {portfolio:,.0f} INR"
        )
        await self.send_message(msg)
