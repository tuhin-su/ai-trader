"""
cleanup.py
==========
Clears temporary data, logs, and models to reset the training environment.
"""
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DIRS_TO_CLEAR = [
    BASE_DIR / "data" / "models",
    BASE_DIR / "data" / "tb_logs",
    BASE_DIR / "data" / "csv",
]

def cleanup():
    print("🧹 Starting cleanup...")
    
    confirm = input("This will delete all models, logs, and CSV data. Proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Cleanup aborted.")
        return

    for d in DIRS_TO_CLEAR:
        if d.exists():
            print(f"Clearing {d}...")
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            # Add a .gitkeep if needed or just leave empty
            (d / ".gitkeep").touch()
    
    # Also clear the database if requested
    db_file = BASE_DIR / "data" / "trading.db"
    if db_file.exists():
        confirm_db = input("Delete trading.db as well? (y/n): ")
        if confirm_db.lower() == 'y':
            db_file.unlink()
            print("Deleted trading.db")

    print("✅ Cleanup complete.")

if __name__ == "__main__":
    cleanup()
