import os
from flask import Flask
from flask_socketio import SocketIO
from config import Config
from models import db
from utils.logger import setup_logger

# Setup logger: tampil di terminal dan file data/logs/app.log
log = setup_logger("app")

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
# eventlet + threading.Thread untuk training YOLO = training tidak pernah jalan (CPU 0%).
# Gunakan threading agar job berat berjalan di OS thread nyata.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Ensure data directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROJECTS_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODELS_FOLDER"], exist_ok=True)

from routes import dashboard, projects, upload, annotation, training, inference
app.register_blueprint(dashboard.bp, url_prefix="/")
app.register_blueprint(projects.bp, url_prefix="/projects")
app.register_blueprint(upload.bp, url_prefix="/projects")
app.register_blueprint(annotation.bp, url_prefix="/projects")
app.register_blueprint(training.bp, url_prefix="/projects")
app.register_blueprint(inference.bp, url_prefix="/projects")

def _migrate_trained_model_columns():
    """Tambah kolom progress pada DB lama (SQLite)."""
    from sqlalchemy import text
    cols = [
        ("progress_epoch", "INTEGER"),
        ("progress_total", "INTEGER"),
        ("progress_message", "TEXT"),
    ]
    for col, ddl in cols:
        try:
            db.session.execute(text(f"ALTER TABLE trained_model ADD COLUMN {col} {ddl}"))
            db.session.commit()
        except Exception:
            db.session.rollback()


with app.app_context():
    db.create_all()
    _migrate_trained_model_columns()

log.info("Flask app initialized | Upload: %s | Projects: %s", app.config["UPLOAD_FOLDER"], app.config["PROJECTS_FOLDER"])

if __name__ == "__main__":
    log.info("Starting server on 0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
