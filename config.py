import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
PROJECTS_FOLDER = BASE_DIR / "data" / "projects"
MODELS_FOLDER = BASE_DIR / "data" / "models"
DATABASE_PATH = BASE_DIR / "data" / "app.db"

# Create directories
for folder in [UPLOAD_FOLDER, PROJECTS_FOLDER, MODELS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SQLite + background training thread (Flask request thread vs worker thread)
    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {"check_same_thread": False},
        "pool_pre_ping": True,
    }
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    PROJECTS_FOLDER = str(PROJECTS_FOLDER)
    MODELS_FOLDER = str(MODELS_FOLDER)
