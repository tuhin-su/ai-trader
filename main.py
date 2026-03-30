"""
AI Trading System — Main Entry Point  (full version)
=====================================================
  ENABLE_TRAINING=True , ENABLE_LIVE_TRADING=False  →  simulation / RL training
  ENABLE_TRAINING=False, ENABLE_LIVE_TRADING=True   →  live trading (Angel One)
"""
import asyncio
import logging

from config.settings import ENABLE_TRAINING, ENABLE_LIVE_TRADING
from utils.console import banner, log_info, log_warn, log_error

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

def _safety_check() -> None:
    if ENABLE_TRAINING and ENABLE_LIVE_TRADING:
        raise RuntimeError(
            "CRITICAL SAFETY VIOLATION: both ENABLE_TRAINING and ENABLE_LIVE_TRADING "
            "are True. Set exactly ONE to True in config/settings.py.")
    if not ENABLE_TRAINING and not ENABLE_LIVE_TRADING:
        raise RuntimeError("Neither mode is True. Set exactly one in config/settings.py.")

async def run_training_pipeline() -> None:
    from training.trainer import Trainer
    from services.database import Database
    log_info("MODE: TRAINING  (simulation only — no real orders)")
    db = Database(); await db.init()
    trainer = Trainer(db=db)
    await trainer.run()
    log_info("Training complete — model saved to data/models/rl_model.zip")
    log_info("Next: python -m utils.backtester")

async def run_live_pipeline() -> None:
    from services.angel_api import AngelAPI
    from services.scanner import Scanner
    from services.execution import ExecutionEngine
    from services.risk_manager import RiskManager
    from services.database import Database
    from services.telegram_service import TelegramService
    from models.model_manager import ModelManager
    from training.continuous_learner import ContinuousLearner
    from utils.dashboard import Dashboard
    from config.settings import SCAN_INTERVAL_SEC

    log_warn("MODE: LIVE TRADING — real orders via Angel One")

    db = Database(); await db.init()
    telegram = TelegramService()
    await telegram.send_message("🚀 AI Trader starting — LIVE MODE")

    api = AngelAPI(); await api.connect()
    model_mgr = ModelManager()
    agent = model_mgr.load_or_raise()

    scanner = Scanner(api=api)
    risk    = RiskManager(db=db)
    engine  = ExecutionEngine(api=api, db=db, telegram=telegram)

    learner   = ContinuousLearner(model_mgr=model_mgr, db=db)
    dashboard = Dashboard(db=db, mode="LIVE")

    learn_task = asyncio.create_task(learner.run())
    dash_task  = asyncio.create_task(dashboard.run())

    try:
        while True:
            candidates = await scanner.scan()
            log_info(f"Scanner: {len(candidates)} candidate(s)")
            await asyncio.gather(*[
                _process_candidate(c, agent, risk, engine, dashboard)
                for c in candidates
            ])
            await asyncio.sleep(SCAN_INTERVAL_SEC)
    except KeyboardInterrupt:
        log_warn("Shutting down…")
    except Exception as exc:
        log_error(f"Unhandled exception: {exc}")
        await telegram.send_message(f"❌ AI Trader crashed: {exc}")
        raise
    finally:
        learn_task.cancel(); dash_task.cancel()
        await telegram.send_message("⏹ AI Trader stopped")
        await db.close()

async def _process_candidate(candidate, agent, risk, engine, dashboard=None) -> None:
    try:
        action, confidence = agent.predict(candidate.features)
        if dashboard: dashboard.log_decision(candidate.symbol, action, confidence)
        if action == 0: return
        if not await risk.check(candidate.symbol, action, confidence): return
        await engine.execute(candidate.symbol, action, confidence, price=candidate.price)
    except Exception as exc:
        log_error(f"Error processing {candidate.symbol}: {exc}")

async def main() -> None:
    banner()
    _safety_check()
    if ENABLE_TRAINING:
        await run_training_pipeline()
    else:
        await run_live_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
