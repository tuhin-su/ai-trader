"""
restore.py
==========
Restores a model and configuration from a backup zip.
"""
import zipfile
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKUP_DIR = BASE_DIR / "backups"
MODELS_DIR = BASE_DIR / "data" / "models"
ENV_FILE = BASE_DIR / ".env"

def restore():
    if not BACKUP_DIR.exists():
        print("❌ No backups directory found.")
        return
    
    backups = sorted(list(BACKUP_DIR.glob("*.zip")), reverse=True)
    if not backups:
        print("❌ No backup files found.")
        return

    print("\nAvailable Backups:")
    for i, b in enumerate(backups):
        print(f"[{i}] {b.name}")
    
    try:
        choice = int(input("\nSelect backup to restore (index): "))
        selected_backup = backups[choice]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    confirm = input(f"Restore {selected_backup.name}? This will OVERWRITE current models. (y/n): ")
    if confirm.lower() != 'y':
        print("Restore aborted.")
        return

    print(f"🔄 Restoring {selected_backup.name}...")
    
    with zipfile.ZipFile(selected_backup, 'r') as zipf:
        # Extract models
        if MODELS_DIR.exists():
            shutil.rmtree(MODELS_DIR)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for member in zipf.namelist():
            if member.startswith("models/"):
                zipf.extract(member, path=BASE_DIR / "data")
        
        # Restore .env if selected
        if ".env.backup" in zipf.namelist():
            restore_env = input("Restore .env file as well? (y/n): ")
            if restore_env.lower() == 'y':
                zipf.extract(".env.backup", path=BASE_DIR)
                shutil.move(BASE_DIR / ".env.backup", ENV_FILE)
                print("Restored .env")

    print("✅ Restore complete.")

if __name__ == "__main__":
    restore()
