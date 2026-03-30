"""
utils/dashboard.py
==================
Real-time terminal dashboard using Rich.
Displays portfolio, open positions, trade history, AI decisions,
and system health in a refreshing TUI.

Run standalone:  python -m utils.dashboard
Or import:       from utils.dashboard import Dashboard
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import List, Dict

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── Colour scheme ────────────────────────────────────────────
C_GREEN  = "bright_green"
C_RED    = "bright_red"
C_YELLOW = "yellow"
C_CYAN   = "cyan"
C_WHITE  = "white"
C_DIM    = "dim white"


class Dashboard:
    """
    Live terminal dashboard.

    Usage:
        db_service = Database(); await db_service.init()
        dash = Dashboard(db=db_service)
        await dash.run()
    """

    def __init__(self, db=None, mode: str = "TRAINING") -> None:
        self.db   = db
        self.mode = mode
        self._trades: List[Dict] = []
        self._positions: List[Dict] = []
        self._portfolio = 100_000.0
        self._daily_pnl = 0.0
        self._ai_log: List[str] = []

    # ── Public ───────────────────────────────────────────────

    async def run(self, refresh_rate: float = 2.0) -> None:
        layout = self._build_layout()
        with Live(layout, refresh_per_second=1 / refresh_rate, screen=True):
            while True:
                await self._refresh_data()
                self._update_layout(layout)
                await asyncio.sleep(refresh_rate)

    def log_decision(self, symbol: str, action: int, confidence: float) -> None:
        action_str = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "?")
        color = {"BUY": C_GREEN, "SELL": C_RED, "HOLD": C_DIM}.get(action_str, C_WHITE)
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{C_DIM}]{ts}[/{C_DIM}] [{color}]{action_str:4s}[/{color}] {symbol:12s} conf={confidence:.0%}"
        self._ai_log.insert(0, entry)
        self._ai_log = self._ai_log[:20]   # keep latest 20

    # ── Layout builders ──────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left",  ratio=2),
            Layout(name="right", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="portfolio", size=10),
            Layout(name="positions"),
        )
        layout["right"].split_column(
            Layout(name="trades"),
            Layout(name="ai_log"),
        )
        return layout

    def _update_layout(self, layout: Layout) -> None:
        layout["header"].update(self._header_panel())
        layout["portfolio"].update(self._portfolio_panel())
        layout["positions"].update(self._positions_panel())
        layout["trades"].update(self._trades_panel())
        layout["ai_log"].update(self._ai_log_panel())
        layout["footer"].update(self._footer_panel())

    # ── Panels ───────────────────────────────────────────────

    def _header_panel(self) -> Panel:
        mode_color = C_YELLOW if self.mode == "TRAINING" else C_RED
        mode_label = f"[{mode_color}][SIMULATION — NO REAL ORDERS][/{mode_color}]" \
                     if self.mode == "TRAINING" \
                     else f"[{C_RED}]⚡ LIVE TRADING — REAL ORDERS[/{C_RED}]"
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        txt = Text.assemble(
            ("  AI TRADING SYSTEM  ", f"bold {C_CYAN}"),
            ("│  ", C_DIM),
            (mode_label, ""),
            ("  │  ", C_DIM),
            (ts, C_DIM),
        )
        return Panel(txt, box=box.HORIZONTALS, style="bold")

    def _portfolio_panel(self) -> Panel:
        pnl_color = C_GREEN if self._daily_pnl >= 0 else C_RED
        pnl_sign  = "+" if self._daily_pnl >= 0 else ""
        pct       = self._daily_pnl / 100_000 * 100

        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(style=C_DIM,   width=20)
        table.add_column(style="bold",  width=20)

        table.add_row("Portfolio Value",
                      f"[{C_WHITE}]₹ {self._portfolio:>12,.0f}[/{C_WHITE}]")
        table.add_row("Daily P&L",
                      f"[{pnl_color}]{pnl_sign}₹ {self._daily_pnl:>10,.0f}  ({pnl_sign}{pct:.2f}%)[/{pnl_color}]")
        table.add_row("Open Positions", str(len(self._positions)))

        return Panel(table, title="[bold]Portfolio[/bold]", box=box.ROUNDED)

    def _positions_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, header_style=f"bold {C_CYAN}", expand=True)
        table.add_column("Symbol",  width=12)
        table.add_column("Qty",     justify="right", width=8)
        table.add_column("Entry",   justify="right", width=10)
        table.add_column("LTP",     justify="right", width=10)
        table.add_column("P&L",     justify="right", width=12)

        for pos in self._positions[:8]:
            pnl = pos.get("pnl", 0.0)
            clr = C_GREEN if pnl >= 0 else C_RED
            table.add_row(
                pos["symbol"],
                str(pos["qty"]),
                f"₹{pos['entry']:,.2f}",
                f"₹{pos.get('ltp', pos['entry']):,.2f}",
                f"[{clr}]{'+' if pnl >= 0 else ''}{pnl:,.0f}[/{clr}]",
            )
        if not self._positions:
            table.add_row("[dim]—[/dim]", "", "", "", "")

        return Panel(table, title="[bold]Open Positions[/bold]", box=box.ROUNDED)

    def _trades_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, header_style=f"bold {C_CYAN}", expand=True)
        table.add_column("Time",    width=10)
        table.add_column("Symbol",  width=12)
        table.add_column("Action",  width=8)
        table.add_column("Price",   justify="right", width=12)
        table.add_column("P&L",     justify="right", width=12)
        table.add_column("Mode",    width=10)

        for t in self._trades[:12]:
            action = t.get("action", "")
            clr    = C_GREEN if action == "BUY" else C_RED
            pnl    = t.get("pnl", 0.0)
            pc     = C_GREEN if pnl >= 0 else C_RED
            table.add_row(
                t.get("time", "—"),
                t.get("symbol", "—"),
                f"[{clr}]{action}[/{clr}]",
                f"₹{t.get('price', 0):,.2f}",
                f"[{pc}]{'+' if pnl >= 0 else ''}{pnl:,.0f}[/{pc}]",
                f"[dim]{t.get('mode', 'sim')}[/dim]",
            )
        if not self._trades:
            table.add_row("[dim]No trades yet[/dim]", "", "", "", "", "")

        return Panel(table, title="[bold]Recent Trades[/bold]", box=box.ROUNDED)

    def _ai_log_panel(self) -> Panel:
        lines = "\n".join(self._ai_log) if self._ai_log else "[dim]Waiting for AI decisions…[/dim]"
        return Panel(
            Text.from_markup(lines),
            title="[bold]AI Decision Log[/bold]",
            box=box.ROUNDED,
        )

    def _footer_panel(self) -> Panel:
        txt = Text.assemble(
            ("  Press ", C_DIM), ("Ctrl+C", f"bold {C_YELLOW}"), (" to stop  │  ", C_DIM),
            ("Risk: MAX 2%/trade · 5%/day · 3 open  │  ", C_DIM),
            ("Powered by Stable Baselines3 + Angel One", C_DIM),
        )
        return Panel(txt, box=box.HORIZONTALS)

    # ── Data refresh ─────────────────────────────────────────

    async def _refresh_data(self) -> None:
        if self.db is None:
            self._mock_refresh()
            return
        try:
            from datetime import date
            self._daily_pnl = await self.db.get_daily_pnl(date.today())
        except Exception:
            pass

    def _mock_refresh(self) -> None:
        """Generates realistic-looking dummy data for standalone demo."""
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        if random.random() < 0.15:
            action = random.choice([1, 2])
            sym    = random.choice(symbols)
            conf   = random.uniform(0.55, 0.95)
            self.log_decision(sym, action, conf)

            price = random.uniform(1500, 3500)
            pnl   = random.uniform(-3000, 5000)
            self._trades.insert(0, {
                "time":   datetime.now().strftime("%H:%M:%S"),
                "symbol": sym,
                "action": {1: "BUY", 2: "SELL"}[action],
                "price":  price,
                "pnl":    pnl,
                "mode":   self.mode.lower(),
            })
            self._trades = self._trades[:20]
            self._daily_pnl += pnl * 0.1
            self._portfolio  = 100_000 + self._daily_pnl

        # Refresh mock positions
        if not self._positions or random.random() < 0.05:
            self._positions = [
                {"symbol": s, "qty": random.randint(5, 50),
                 "entry": random.uniform(1500, 3500),
                 "ltp":   random.uniform(1400, 3600),
                 "pnl":   random.uniform(-2000, 4000)}
                for s in random.sample(symbols, random.randint(0, 3))
            ]


# ─────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = "LIVE" if "--live" in sys.argv else "TRAINING"
    dash = Dashboard(mode=mode)
    try:
        asyncio.run(dash.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
