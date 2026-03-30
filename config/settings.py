"""
config/settings.py
==================
Central configuration.  Edit this file to switch modes.

CRITICAL: Never set both ENABLE_TRAINING and ENABLE_LIVE_TRADING to True.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Mode control ────────────────────────────────────────────
# Set exactly ONE of these to True.
ENABLE_TRAINING: bool = True       # ← simulation / RL training
ENABLE_LIVE_TRADING: bool = False  # ← real Angel One orders

# ── Capital & risk ──────────────────────────────────────────
CAPITAL: float = 100_000.0         # Starting capital (INR)
MAX_RISK_PER_TRADE: float = 0.02   # 2 % of capital per trade
MAX_DAILY_LOSS: float = 0.05       # 5 % daily loss limit
STOP_LOSS_PCT: float = 0.01        # 1 % stop-loss per position
TARGET_PCT: float = 0.02           # 2 % take-profit per position
MAX_OPEN_TRADES: int = 3

# ── Scanner ─────────────────────────────────────────────────
SCAN_LIMIT: int = 50               # Max stocks evaluated per cycle
SCAN_INTERVAL_SEC: int = 60        # Seconds between scans (live mode)

# ── Model ───────────────────────────────────────────────────
MODEL_PATH: str = str(BASE_DIR / "data" / "models" / "rl_model")
MODEL_ALGO: str = "PPO"            # "PPO" | "SAC" | "A2C"
TRAIN_TIMESTEPS: int = 200_000
RETRAIN_INTERVAL_TRADES: int = 100 # Retrain after N new live trades

# ── Angel One API ───────────────────────────────────────────
ANGEL_API_KEY: str = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID: str = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD: str = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET: str = os.getenv("ANGEL_TOTP_SECRET", "")

# ── Telegram ────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Database ────────────────────────────────────────────────
DB_PATH: str = str(BASE_DIR / "data" / "trading.db")

# ── Indicators ──────────────────────────────────────────────
EMA_SHORT: int = 9
EMA_LONG: int = 21
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BOLLINGER_PERIOD: int = 20
BOLLINGER_STD: float = 2.0

# ── RL Environment ──────────────────────────────────────────
LOOKBACK_WINDOW: int = 30          # Candles fed to the state encoder
REWARD_SCALE: float = 100.0        # Scale raw P&L reward
DRAWDOWN_PENALTY: float = 2.0      # Multiplier on drawdown in reward
RISK_PENALTY: float = 1.5          # Multiplier on risk in reward
