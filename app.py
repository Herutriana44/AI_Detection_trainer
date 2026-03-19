import os
from flask import Flask
from flask_socketio import SocketIO
from config import Config
from models import db

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

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

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
