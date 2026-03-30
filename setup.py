#!/usr/bin/env python3
"""
setup.py  — one-shot project bootstrap
=======================================
Run once after cloning:  python setup.py

What it does:
  1. Creates all required directories
  2. Copies .env.example → .env  (if .env doesn't exist)
  3. Creates data/stocks.json with the default watchlist
  4. Verifies Python version (≥ 3.10)
  5. Checks that required packages are importable
  6. Prints a getting-started checklist
"""
import sys
import os
import shutil
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIRS = [
    "config", "models", "env", "services",
    "training", "utils", "tests",
    "data/models", "data/csv", "data/tb_logs",
]

DEFAULT_WATCHLIST = {
    "watchlist": [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "KOTAKBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC",
        "BAJFINANCE", "ASIANPAINT", "MARUTI", "TITAN", "LT",
        "AXISBANK", "HCLTECH", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
    ]
}

OK   = "\033[92m✓\033[0m"
WARN = "\033[93m!\033[0m"
ERR  = "\033[91m✗\033[0m"
BOLD = "\033[1m"
RST  = "\033[0m"


def step(msg: str) -> None:
    print(f"  {msg}")


def main() -> None:
    print(f"\n{BOLD}AI Trading System — Project Setup{RST}\n")

    # 1. Python version
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"  {ERR}  Python 3.10+ required (found {major}.{minor})")
        sys.exit(1)
    step(f"{OK}  Python {major}.{minor}")

    # 2. Directories
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        # Touch __init__.py for Python packages
        if not d.startswith("data"):
            init = ROOT / d / "__init__.py"
            if not init.exists():
                init.touch()
    step(f"{OK}  Directories created")

    # 3. .env file
    env_file    = ROOT / ".env"
    env_example = ROOT / ".env.example"
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            step(f"{WARN}  .env created from .env.example — fill in your credentials!")
        else:
            env_file.write_text(
                "ANGEL_API_KEY=\nANGEL_CLIENT_ID=\nANGEL_PASSWORD=\n"
                "ANGEL_TOTP_SECRET=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n"
            )
            step(f"{WARN}  .env created (empty) — fill in your credentials!")
    else:
        step(f"{OK}  .env already exists")

    # 4. stocks.json
    stocks_file = ROOT / "data" / "stocks.json"
    if not stocks_file.exists():
        stocks_file.write_text(json.dumps(DEFAULT_WATCHLIST, indent=2))
        step(f"{OK}  data/stocks.json created with {len(DEFAULT_WATCHLIST['watchlist'])} symbols")
    else:
        step(f"{OK}  data/stocks.json already exists")

    # 5. Package checks
    _check_packages()

    # 6. Checklist
    print(f"\n{BOLD}Getting Started{RST}")
    checklist = [
        ("Edit .env",                     "Add your Angel One API key, client ID, password, TOTP secret"),
        ("pip install -r requirements.txt", "Install all dependencies"),
        ("ENABLE_TRAINING = True",          "Confirm in config/settings.py (default)"),
        ("python main.py",                  "Run training — generates data/models/rl_model.zip"),
        ("python -m utils.backtester",      "Evaluate on out-of-sample data"),
        ("python -m utils.dashboard",       "Watch the live dashboard (demo mode)"),
        ("Flip modes in settings.py",       "Set ENABLE_TRAINING=False, ENABLE_LIVE_TRADING=True for live"),
        ("pytest tests/test_suite.py -v",   "Run full test suite before going live"),
    ]
    for i, (cmd, desc) in enumerate(checklist, 1):
        print(f"  {i}. {BOLD}{cmd}{RST}")
        print(f"     {desc}")
    print()


def _check_packages() -> None:
    checks = [
        ("numpy",              "numpy"),
        ("pandas",             "pandas"),
        ("gymnasium",          "gymnasium"),
        ("stable_baselines3",  "stable-baselines3"),
        ("torch",              "torch"),
        ("aiosqlite",          "aiosqlite"),
        ("aiohttp",            "aiohttp"),
        ("dotenv",             "python-dotenv"),
    ]
    missing = []
    for mod, pkg in checks:
        try:
            __import__(mod)
            step(f"{OK}  {pkg}")
        except ImportError:
            step(f"{WARN}  {pkg} not installed")
            missing.append(pkg)
    if missing:
        print(f"\n  {WARN}  Run:  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
