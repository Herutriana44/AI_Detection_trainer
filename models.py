from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    images = db.relationship("Image", backref="project", cascade="all, delete-orphan")
    trained_models = db.relationship("TrainedModel", backref="project", cascade="all, delete-orphan")

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    annotated = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    annotations = db.relationship("Annotation", backref="image", cascade="all, delete-orphan")

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image.id"), nullable=False)
    class_id = db.Column(db.Integer, nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    x_center = db.Column(db.Float, nullable=False)  # normalized 0-1
    y_center = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)

class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    base_model = db.Column(db.String(50), nullable=False)  # yolov8n, yolov8s, etc
    model_path = db.Column(db.String(500))
    epochs = db.Column(db.Integer)
    status = db.Column(db.String(50), default="pending")  # pending, preparing, training, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Progress (diisi saat training berjalan)
    progress_epoch = db.Column(db.Integer, default=0)
    progress_total = db.Column(db.Integer)
    progress_message = db.Column(db.Text)
