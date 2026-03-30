"""
utils/backtester.py
====================
Walk-forward backtester for evaluating a trained RL model on unseen data.

Usage:
    python -m utils.backtester --symbol RELIANCE --days 180

Outputs:
    • Equity curve (terminal chart via Rich)
    • Sharpe ratio, max drawdown, win rate, total return
    • Per-trade breakdown
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Backtester:
    """Walk-forward single-symbol backtest against a saved RL model."""

    def __init__(self, symbol: str = "RELIANCE", days: int = 180) -> None:
        self.symbol = symbol
        self.days   = days

    async def run(self) -> dict:
        from training.dataset_builder import DatasetBuilder
        from utils.indicators import Indicators
        from models.model_manager import ModelManager
        from env.trading_env import TradingEnv

        # 1. Load data
        builder = DatasetBuilder()
        _, val_data = await builder.build()
        logger.info(f"Backtesting on {len(val_data)} out-of-sample rows")

        # 2. Load model
        mgr   = ModelManager()
        agent = mgr.load_or_raise()

        # 3. Run episode
        env = TradingEnv(data=val_data)
        obs, _ = env.reset()
        done   = False

        equity_curve: List[float] = [env.initial_capital]
        trade_log: List[dict]     = []
        step = 0

        while not done:
            action, confidence = agent.predict(obs)
            prev_price = float(env.data[env.current_step - 1, 3])
            obs, reward, done, _, info = env.step(action)
            curr_price = float(env.data[env.current_step - 1, 3])
            equity_curve.append(info["portfolio_value"])

            if action != 0:
                trade_log.append({
                    "step":       step,
                    "action":     {1: "BUY", 2: "SELL"}.get(action, "?"),
                    "price":      curr_price,
                    "confidence": confidence,
                    "portfolio":  info["portfolio_value"],
                })
            step += 1

        metrics = self._compute_metrics(equity_curve, trade_log, env.initial_capital)
        self._print_report(metrics, equity_curve, trade_log)
        return metrics

    # ── Metrics ──────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(equity: List[float], trades: List[dict], initial: float) -> dict:
        arr = np.array(equity)
        returns = np.diff(arr) / (arr[:-1] + 1e-9)

        total_return = (arr[-1] - initial) / initial
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) \
                 if len(returns) > 1 else 0.0

        peak, max_dd = initial, 0.0
        for v in arr:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / (peak + 1e-9))

        sells = [t for t in trades if t["action"] == "SELL"]
        buys  = [t for t in trades if t["action"] == "BUY"]

        # Pair buys and sells naively
        wins = 0
        for b, s in zip(buys, sells):
            if s["price"] > b["price"]:
                wins += 1
        pairs = min(len(buys), len(sells))
        win_rate = wins / pairs if pairs > 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate":     win_rate,
            "total_trades": len(trades),
            "final_portfolio": arr[-1],
        }

    # ── Terminal report ──────────────────────────────────────

    @staticmethod
    def _print_report(metrics: dict, equity: List[float], trades: List[dict]) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box as rbox

            c = Console()
            c.print("\n[bold cyan]═══ Backtest Report ═══[/bold cyan]\n")

            t = Table(box=rbox.SIMPLE, show_header=False)
            t.add_column(style="dim", width=22)
            t.add_column(style="bold", width=18)

            tr_c = "bright_green" if metrics["total_return"] >= 0 else "bright_red"
            t.add_row("Total Return",   f"[{tr_c}]{metrics['total_return']:+.2%}[/{tr_c}]")
            t.add_row("Sharpe Ratio",   f"{metrics['sharpe_ratio']:.3f}")
            t.add_row("Max Drawdown",   f"[bright_red]{metrics['max_drawdown']:.2%}[/bright_red]")
            t.add_row("Win Rate",       f"{metrics['win_rate']:.1%}")
            t.add_row("Total Trades",   str(metrics["total_trades"]))
            t.add_row("Final Portfolio", f"₹{metrics['final_portfolio']:,.0f}")
            c.print(t)

            # Mini ASCII equity curve
            c.print("[dim]Equity curve (normalised)[/dim]")
            arr = np.array(equity)
            if arr.max() > arr.min():
                norm = (arr - arr.min()) / (arr.max() - arr.min())
                WIDTH, HEIGHT = 60, 8
                cols = np.array_split(norm, WIDTH)
                rows = []
                for row in range(HEIGHT - 1, -1, -1):
                    threshold = row / (HEIGHT - 1)
                    line = "".join("█" if col.mean() >= threshold else " " for col in cols)
                    rows.append(line)
                color = "bright_green" if metrics["total_return"] >= 0 else "bright_red"
                for row in rows:
                    c.print(f"[{color}]│{row}│[/{color}]")
                c.print(f"[dim]  ▲ ₹{arr.min():,.0f}  →  ₹{arr.max():,.0f}[/dim]\n")
        except ImportError:
            print(f"Backtest complete: {metrics}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trader — Backtester")
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--days",   type=int, default=180)
    args = parser.parse_args()

    bt = Backtester(symbol=args.symbol, days=args.days)
    asyncio.run(bt.run())
