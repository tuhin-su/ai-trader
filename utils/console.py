"""
utils/console.py
================
Coloured terminal output helpers.
"""
import logging

logger = logging.getLogger("console")

_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"


def banner() -> None:
    print(f"""
{_CYAN}{_BOLD}
╔══════════════════════════════════════════════════╗
║        AI TRADING SYSTEM  —  RL ENGINE           ║
║        Angel One  |  NSE  |  Python 3.11         ║
╚══════════════════════════════════════════════════╝
{_RESET}""")


def log_info(msg: str) -> None:
    print(f"{_GREEN}[INFO]{_RESET}  {msg}")
    logger.info(msg)


def log_warn(msg: str) -> None:
    print(f"{_YELLOW}[WARN]{_RESET}  {msg}")
    logger.warning(msg)


def log_error(msg: str) -> None:
    print(f"{_RED}[ERR ]{_RESET}  {msg}")
    logger.error(msg)
