"""
backup.py
=========
Archives current models and configuration into a timestamped zip.
"""
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKUP_DIR = BASE_DIR / "backups"
MODELS_DIR = BASE_DIR / "data" / "models"
ENV_FILE = BASE_DIR / ".env"

def backup():
    BACKUP_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"ai_trader_backup_{timestamp}.zip"
    
    print(f"📦 Creating backup: {backup_path.name}...")
    
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Backup models
        if MODELS_DIR.exists():
            for file in MODELS_DIR.rglob('*'):
                if file.is_file():
                    zipf.write(file, arcname=Path("models") / file.relative_to(MODELS_DIR))
        
        # Backup .env
        if ENV_FILE.exists():
            zipf.write(ENV_FILE, arcname=".env.backup")

    print(f"✅ Backup created successfully at {backup_path}")

if __name__ == "__main__":
    backup()
