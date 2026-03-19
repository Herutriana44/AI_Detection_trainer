import os
import threading
from pathlib import Path
from flask import Blueprint, request, render_template, jsonify, send_file
from models import db, Project, Image, Annotation, TrainedModel
from utils.logger import get_logger

log = get_logger("training")
bp = Blueprint("training", __name__)

# Store training jobs
training_jobs = {}

YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

def get_socketio():
    from app import socketio
    return socketio

def get_classes_from_project(project):
    classes = set()
    for img in project.images:
        for ann in img.annotations:
            classes.add((ann.class_id, ann.class_name))
    return sorted(list(classes), key=lambda x: x[0])

def run_training(app, project_id, model_id):
    job_id = f"{project_id}_{model_id}"
    log.info("Training dimulai | project_id=%s model_id=%s", project_id, model_id)
    with app.app_context():
        model_record = TrainedModel.query.get(model_id)
        if not model_record or model_record.project_id != project_id:
            log.warning("Model record tidak ditemukan atau tidak cocok | project_id=%s model_id=%s", project_id, model_id)
            return
        
        project = Project.query.get(project_id)
        annotated_images = [img for img in project.images if img.annotated]
        if not annotated_images:
            log.warning("Tidak ada gambar teranotasi untuk project_id=%s", project_id)
            return
        
        classes = get_classes_from_project(project)
        class_names = [c[1] for c in classes]
        
        project_dir = Path(app.config["PROJECTS_FOLDER"]) / str(project_id)
        images_dir = project_dir / "images"
        dataset_dir = project_dir / "dataset"
        dataset_dir.mkdir(exist_ok=True)
    
        train_images = dataset_dir / "images" / "train"
        val_images = dataset_dir / "images" / "val"
        train_labels = dataset_dir / "labels" / "train"
        val_labels = dataset_dir / "labels" / "val"
        for d in [train_images, val_images, train_labels, val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        import shutil
        
        n = len(annotated_images)
        split = int(n * 0.8)
        train_imgs = annotated_images[:split]
        val_imgs = annotated_images[split:]
        
        def copy_and_create_labels(imgs, img_dst, lbl_dst):
            for img in imgs:
                src = Path(app.config["PROJECTS_FOLDER"]) / img.filepath
                if not src.exists():
                    continue
                dst_name = f"{img.id}.{img.filename.rsplit('.', 1)[-1]}"
                shutil.copy(src, img_dst / dst_name)
                lbl_path = lbl_dst / f"{img.id}.txt"
                with open(lbl_path, "w") as f:
                    for ann in img.annotations:
                        f.write(f"{ann.class_id} {ann.x_center} {ann.y_center} {ann.width} {ann.height}\n")
        
        copy_and_create_labels(train_imgs, train_images, train_labels)
        copy_and_create_labels(val_imgs, val_images, val_labels)
        log.info("Dataset siap | train=%d val=%d classes=%s", len(train_imgs), len(val_imgs), class_names)
        
        data_yaml = f"""
path: {dataset_dir.absolute()}
train: images/train
val: images/val
names:
"""
        for i, name in enumerate(class_names):
            data_yaml += f"  {i}: {name}\n"
        
        (dataset_dir / "data.yaml").write_text(data_yaml)
        
        model_record.status = "training"
        db.session.commit()
        
        log_file = project_dir / "runs" / f"train_{model_id}_log.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_record.base_model)
            log.info("YOLO model loaded | base=%s epochs=%d", model_record.base_model, model_record.epochs or 50)
            
            results = model.train(
                data=str(dataset_dir / "data.yaml"),
                epochs=model_record.epochs or 50,
                imgsz=640,
                project=str(project_dir / "runs"),
                name=f"train_{model_id}",
                exist_ok=True,
                verbose=True
            )
            
            best_path = Path(results.save_dir) / "weights" / "best.pt"
            if best_path.exists():
                model_dir = Path(app.config["MODELS_FOLDER"]) / str(project_id)
                model_dir.mkdir(exist_ok=True)
                dest = model_dir / f"best_{model_id}.pt"
                shutil.copy(best_path, dest)
                model_record.model_path = str(dest)
                log.info("Training selesai | model disimpan ke %s", dest)
            else:
                log.warning("File best.pt tidak ditemukan di %s", results.save_dir)
            
            model_record.status = "completed"
        except Exception as e:
            log.exception("Training gagal | project_id=%s model_id=%s: %s", project_id, model_id, e)
            model_record.status = "failed"
            model_record.model_path = str(e)
        finally:
            db.session.commit()
            training_jobs.pop(job_id, None)

@bp.route("/<int:project_id>/training")
def training_page(project_id):
    project = Project.query.get_or_404(project_id)
    annotated_count = sum(1 for img in project.images if img.annotated)
    models = project.trained_models
    return render_template(
        "training.html",
        project=project,
        annotated_count=annotated_count,
        models=models,
        yolo_models=YOLO_MODELS
    )

@bp.route("/<int:project_id>/training/start", methods=["POST"])
def start_training(project_id):
    project = Project.query.get_or_404(project_id)
    annotated_count = sum(1 for img in project.images if img.annotated)
    if annotated_count == 0:
        return jsonify({"error": "Belum ada dataset yang teranotasi."}), 400
    
    base_model = request.json.get("base_model", "yolov8n.pt")
    epochs = int(request.json.get("epochs", 50))
    
    model_record = TrainedModel(
        project_id=project_id,
        base_model=base_model,
        epochs=epochs,
        status="pending"
    )
    db.session.add(model_record)
    db.session.commit()
    
    def run():
        run_training(request.app, project_id, model_record.id)
    
    thread = threading.Thread(target=run)
    thread.start()
    training_jobs[f"{project_id}_{model_record.id}"] = thread
    log.info("Training job dijalankan | project_id=%s model_id=%s base=%s epochs=%d", project_id, model_record.id, base_model, epochs)
    
    return jsonify({"model_id": model_record.id, "success": True})

@bp.route("/<int:project_id>/training/<int:model_id>/status")
def training_status(project_id, model_id):
    model = TrainedModel.query.filter_by(id=model_id, project_id=project_id).first_or_404()
    return jsonify({
        "status": model.status,
        "model_path": model.model_path if model.status == "completed" else None
    })

@bp.route("/<int:project_id>/training/<int:model_id>/download")
def download_model(project_id, model_id):
    model = TrainedModel.query.filter_by(id=model_id, project_id=project_id).first_or_404()
    if model.status != "completed" or not model.model_path or not os.path.exists(model.model_path):
        return "Model tidak tersedia", 404
    return send_file(
        model.model_path,
        as_attachment=True,
        download_name=f"best_{model.base_model.replace('.pt', '')}.pt"
    )
