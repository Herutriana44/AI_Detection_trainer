import os
import uuid
import base64
import tempfile
from pathlib import Path
from flask import Blueprint, request, render_template, jsonify, send_file
from models import Project, TrainedModel
from utils.logger import get_logger

log = get_logger("inference")
bp = Blueprint("inference", __name__)

@bp.route("/<int:project_id>/inference")
def inference_page(project_id):
    project = Project.query.get_or_404(project_id)
    models = [m for m in project.trained_models if m.status == "completed" and m.model_path and os.path.exists(m.model_path)]
    return render_template("inference.html", project=project, models=models)

@bp.route("/<int:project_id>/inference/detect", methods=["POST"])
def detect(project_id):
    project = Project.query.get_or_404(project_id)
    model_id = request.form.get("model_id", type=int)
    model = TrainedModel.query.filter_by(id=model_id, project_id=project_id).first_or_404()
    if model.status != "completed" or not model.model_path or not os.path.exists(model.model_path):
        return jsonify({"error": "Model tidak tersedia"}), 400
    
    source_type = request.form.get("source_type", "image")
    
    if source_type == "image":
        if "image" not in request.files:
            return jsonify({"error": "Tidak ada gambar"}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "File tidak valid"}), 400
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            file.save(tmp.name)
            result = run_detection(model.model_path, tmp.name)
            os.unlink(tmp.name)
    
    elif source_type == "video":
        if "video" not in request.files:
            return jsonify({"error": "Tidak ada video"}), 400
        file = request.files["video"]
        if not file.filename:
            return jsonify({"error": "File tidak valid"}), 400
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            file.save(tmp.name)
            result = run_detection(model.model_path, tmp.name, is_video=True)
            os.unlink(tmp.name)
    
    elif source_type == "url":
        url = request.form.get("url", "").strip()
        if not url:
            return jsonify({"error": "URL tidak valid"}), 400
        result = run_detection(model.model_path, url, is_video=url.startswith("rtsp://") or "video" in url.lower())
    
    else:
        return jsonify({"error": "Tipe sumber tidak valid"}), 400
    
    return jsonify(result)

def run_detection(model_path, source, is_video=False):
    from ultralytics import YOLO
    import cv2
    
    log.info("Inference dimulai | model=%s source=%s is_video=%s", model_path, str(source)[:80], is_video)
    model = YOLO(model_path)
    results = list(model(source, verbose=False))
    
    output_images = []
    max_frames = 10 if is_video else 1
    
    for i, r in enumerate(results):
        if i >= max_frames:
            break
        img = r.orig_img
        if img is None:
            continue
        annotated = r.plot()
        _, buf = cv2.imencode(".jpg", annotated)
        b64 = base64.b64encode(buf).decode("utf-8")
        output_images.append(b64)
    
    log.info("Inference selesai | frames=%d", len(output_images))
    return {"images": output_images}
