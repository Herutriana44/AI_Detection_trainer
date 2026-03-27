import os
import re
import threading
from pathlib import Path
from flask import Blueprint, request, render_template, jsonify, send_file, current_app
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

def _update_training_progress(app, model_id, project_id, **kwargs):
    """Update baris TrainedModel di thread training (commit terpisah)."""
    with app.app_context():
        mr = TrainedModel.query.filter_by(id=model_id, project_id=project_id).first()
        if not mr:
            return
        for k, v in kwargs.items():
            if hasattr(mr, k) and v is not None:
                setattr(mr, k, v)
        db.session.commit()


def run_training(app, project_id, model_id):
    job_id = f"{project_id}_{model_id}"
    # Backup: pastikan terlihat di Colab/Kaggle meski buffering logging
    print(f"[training] thread started pid={os.getpid()} model_id={model_id}", flush=True)
    log.info("Training dimulai | project_id=%s model_id=%s pid=%s", project_id, model_id, os.getpid())
    with app.app_context():
        model_record = TrainedModel.query.get(model_id)
        if not model_record or model_record.project_id != project_id:
            log.warning("Model record tidak ditemukan atau tidak cocok | project_id=%s model_id=%s", project_id, model_id)
            return
        
        project = Project.query.get(project_id)
        annotated_images = [img for img in project.images if img.annotated]
        if not annotated_images:
            log.warning("Tidak ada gambar teranotasi untuk project_id=%s", project_id)
            model_record.status = "failed"
            model_record.progress_message = "Tidak ada gambar teranotasi."
            model_record.model_path = "Tidak ada gambar teranotasi."
            db.session.commit()
            training_jobs.pop(job_id, None)
            return
        
        total_epochs = model_record.epochs or 50
        # Langsung tampilkan status agar UI/log tidak stuck di "pending" saat dataset besar
        model_record.status = "preparing"
        model_record.progress_epoch = 0
        model_record.progress_total = total_epochs
        model_record.progress_message = "Menyiapkan dataset (copy gambar & label)..."
        db.session.commit()
        log.info("Status: preparing | project_id=%s model_id=%s", project_id, model_id)
        
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
        
        total_copy = len(train_imgs) + len(val_imgs)
        done = [0]

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
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == total_copy:
                    _update_training_progress(
                        app, model_id, project_id,
                        progress_message=f"Menyiapkan dataset: {done[0]}/{total_copy} gambar...",
                    )
        
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
        model_record.progress_message = "Memuat model YOLO & memulai epoch..."
        model_record.progress_epoch = 0
        model_record.progress_total = total_epochs
        db.session.commit()
        
        log_file = project_dir / "runs" / f"train_{model_id}_log.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_record.base_model)
            log.info("YOLO model loaded | base=%s epochs=%d", model_record.base_model, total_epochs)
            
            mid = model_id
            pid = project_id

            def on_train_start(trainer):
                te = int(getattr(trainer, "epochs", total_epochs) or total_epochs)
                log.info("Training YOLO dimulai | epochs=%s", te)
                _update_training_progress(
                    app, mid, pid,
                    status="training",
                    progress_epoch=0,
                    progress_total=te,
                    progress_message=f"Menjalankan training... (0/{te} epoch)",
                )

            def on_train_epoch_end(trainer):
                te = int(getattr(trainer, "epochs", total_epochs) or total_epochs)
                ep = int(getattr(trainer, "epoch", 0))
                # Ultralytics: setelah epoch ke-ep selesai (0-based)
                ep_done = min(ep + 1, te)
                msg = f"Epoch {ep_done}/{te} (train)"
                log.info("Training | %s", msg)
                _update_training_progress(
                    app, mid, pid,
                    progress_epoch=ep_done,
                    progress_total=te,
                    progress_message=msg,
                )

            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            results = model.train(
                data=str(dataset_dir / "data.yaml"),
                epochs=total_epochs,
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
            model_record.progress_total = total_epochs
            model_record.progress_epoch = total_epochs
            model_record.progress_message = "Selesai."
        except Exception as e:
            log.exception("Training gagal | project_id=%s model_id=%s: %s", project_id, model_id, e)
            model_record.status = "failed"
            err_msg = str(e)
            model_record.model_path = err_msg
            model_record.progress_message = f"Error: {err_msg}"
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
    
    app_instance = current_app._get_current_object()
    mid = model_record.id

    def run_training_job():
        run_training(app_instance, project_id, mid)

    # Dengan async_mode=threading (bukan eventlet), OS thread ini benar-benar jalan paralel.
    thread = threading.Thread(target=run_training_job, name="YoloTraining", daemon=False)
    thread.start()
    training_jobs[f"{project_id}_{mid}"] = thread
    log.info("Training job dijadwalkan | project_id=%s model_id=%s base=%s epochs=%d", project_id, mid, base_model, epochs)
    
    return jsonify({"model_id": model_record.id, "success": True})

def _compute_progress_percent(model):
    """Gabungkan fase preparing (copy) + training (epoch) ke 0–100%."""
    msg = model.progress_message or ""
    if model.status == "completed":
        return 100
    if model.status == "failed":
        return 0
    if model.status == "pending":
        return 0
    if model.status == "preparing":
        m = re.search(r"(\d+)\s*/\s*(\d+)\s*gambar", msg)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b > 0:
                return min(45, int(45 * a / b))
        return 5
    if model.status == "training":
        pe = model.progress_epoch or 0
        pt = model.progress_total or model.epochs or 1
        if pt <= 0:
            pt = 1
        # 45–100% untuk fase epoch
        return min(100, 45 + int(55 * pe / pt))
    return 0


@bp.route("/<int:project_id>/training/<int:model_id>/status")
def training_status(project_id, model_id):
    model = TrainedModel.query.filter_by(id=model_id, project_id=project_id).first_or_404()
    pe = model.progress_epoch or 0
    pt = model.progress_total or model.epochs or 0
    err = None
    if model.status == "failed":
        err = model.progress_message or model.model_path
    pct = _compute_progress_percent(model)
    return jsonify({
        "status": model.status,
        "model_path": model.model_path if model.status == "completed" else None,
        "progress_epoch": pe,
        "progress_total": pt,
        "progress_message": model.progress_message,
        "progress_percent": pct,
        "error": err,
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
