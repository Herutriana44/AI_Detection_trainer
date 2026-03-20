import os
import uuid
import tempfile
import zipfile
from pathlib import Path
from flask import Blueprint, request, redirect, url_for, flash, render_template, send_from_directory, current_app, send_file
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
from models import db, Project, Image, Annotation
from utils.logger import get_logger
from utils.dataset_io import (
    IMPORT_FORMATS,
    EXPORT_FORMATS,
    import_yolo,
    import_coco_with_images,
    import_coco_json_only,
    import_voc,
    import_csv,
    export_yolo,
    export_coco,
    export_voc,
)

log = get_logger("upload")
bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/<int:project_id>/image/<path:filepath>")
def serve_image(project_id, filepath):
    Project.query.get_or_404(project_id)
    base = current_app.config["PROJECTS_FOLDER"]
    return send_from_directory(base, filepath)

@bp.route("/<int:project_id>/dataset")
def dataset(project_id):
    project = Project.query.get_or_404(project_id)
    images = project.images
    return render_template(
        "dataset.html",
        project=project,
        images=images,
        import_formats=IMPORT_FORMATS,
        export_formats=EXPORT_FORMATS,
    )

@bp.route("/<int:project_id>/dataset/upload", methods=["POST"])
def upload_files(project_id):
    project = Project.query.get_or_404(project_id)
    project_dir = os.path.join(current_app.config["PROJECTS_FOLDER"], str(project_id), "images")
    os.makedirs(project_dir, exist_ok=True)
    
    if "files" not in request.files and "file" not in request.files:
        flash("Tidak ada file yang diupload.", "error")
        return redirect(url_for("upload.dataset", project_id=project_id))
    
    files = request.files.getlist("files") if "files" in request.files else [request.files["file"]]
    count = 0
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(project_dir, unique_name)
            file.save(filepath)
            
            try:
                with PILImage.open(filepath) as img:
                    w, h = img.size
            except Exception:
                w, h = None, None
            
            image = Image(
                project_id=project_id,
                filename=file.filename,
                filepath=os.path.relpath(filepath, current_app.config["PROJECTS_FOLDER"]),
                width=w,
                height=h
            )
            db.session.add(image)
            count += 1
    
    db.session.commit()
    log.info("Upload berhasil | project_id=%s count=%d", project_id, count)
    flash(f"{count} gambar berhasil diupload.", "success")
    return redirect(url_for("upload.dataset", project_id=project_id))

@bp.route("/<int:project_id>/dataset/import", methods=["POST"])
def import_dataset(project_id):
    project = Project.query.get_or_404(project_id)
    format_type = request.form.get("format", "yolo").lower()
    if "file" not in request.files:
        flash("Pilih file dataset untuk di-import.", "error")
        return redirect(url_for("upload.dataset", project_id=project_id))
    file = request.files["file"]
    if not file or not file.filename:
        flash("File tidak valid.", "error")
        return redirect(url_for("upload.dataset", project_id=project_id))
    projects_folder = current_app.config["PROJECTS_FOLDER"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        filepath = tmp_path / (file.filename or "upload")
        file.save(str(filepath))
        result_images = []
        try:
            if format_type == "yolo" and filepath.suffix.lower() == ".zip":
                _, _, _, result_images = import_yolo(filepath, projects_folder, project_id)
            elif format_type == "coco":
                if filepath.suffix.lower() == ".zip":
                    _, _, _, result_images = import_coco_with_images(filepath, projects_folder, project_id)
                elif filepath.suffix.lower() == ".json":
                    _, _, _, result_images = import_coco_json_only(filepath, projects_folder, project_id)
                else:
                    flash("Format COCO: upload file .zip (gambar+JSON) atau .json", "error")
                    return redirect(url_for("upload.dataset", project_id=project_id))
            elif format_type == "voc" and filepath.suffix.lower() == ".zip":
                _, _, _, result_images = import_voc(filepath, projects_folder, project_id)
            elif format_type == "csv":
                if filepath.suffix.lower() == ".zip":
                    with zipfile.ZipFile(filepath, "r") as zf:
                        zf.extractall(tmp_path)
                    csv_files = list(tmp_path.rglob("*.csv"))
                    if not csv_files:
                        flash("Tidak ada file CSV di dalam zip.", "error")
                        return redirect(url_for("upload.dataset", project_id=project_id))
                    _, _, _, result_images = import_csv(csv_files[0], projects_folder, project_id, tmp_path)
                else:
                    _, _, _, result_images = import_csv(filepath, projects_folder, project_id)
            else:
                flash(f"Format tidak didukung. YOLO/VOC/CSV: .zip, COCO: .zip atau .json", "error")
                return redirect(url_for("upload.dataset", project_id=project_id))
        except Exception as e:
            log.exception("Import dataset gagal: %s", e)
            flash(f"Import gagal: {e}", "error")
            return redirect(url_for("upload.dataset", project_id=project_id))
        img_count = 0
        ann_count = 0
        for r in result_images:
            img = Image(
                project_id=project_id,
                filename=r["filename"],
                filepath=r["filepath"],
                width=r.get("width"),
                height=r.get("height"),
                annotated=len(r.get("annotations", [])) > 0,
            )
            db.session.add(img)
            db.session.flush()
            for ann in r.get("annotations", []):
                a = Annotation(
                    image_id=img.id,
                    class_id=ann["class_id"],
                    class_name=ann["class_name"],
                    x_center=ann["x_center"],
                    y_center=ann["y_center"],
                    width=ann["width"],
                    height=ann["height"],
                )
                db.session.add(a)
                ann_count += 1
            img_count += 1
        db.session.commit()
        log.info("Import dataset berhasil | project_id=%s format=%s images=%d", project_id, format_type, img_count)
        flash(f"Import berhasil: {img_count} gambar, {ann_count} anotasi.", "success")
    return redirect(url_for("upload.dataset", project_id=project_id))


@bp.route("/<int:project_id>/dataset/export/<format_type>")
def export_dataset(project_id, format_type):
    project = Project.query.get_or_404(project_id)
    projects_folder = current_app.config["PROJECTS_FOLDER"]
    models_folder = current_app.config.get("MODELS_FOLDER", "")
    try:
        if format_type == "yolo":
            out_path = export_yolo(project, projects_folder, models_folder)
            return send_file(
                out_path,
                as_attachment=True,
                download_name=f"{project.name}_yolo.zip",
                mimetype="application/zip",
            )
        elif format_type == "coco":
            out_path = export_coco(project, projects_folder)
            return send_file(
                out_path,
                as_attachment=True,
                download_name=f"{project.name}_coco.json",
                mimetype="application/json",
            )
        elif format_type == "voc":
            out_path = export_voc(project, projects_folder)
            return send_file(
                out_path,
                as_attachment=True,
                download_name=f"{project.name}_voc.zip",
                mimetype="application/zip",
            )
        else:
            flash("Format export tidak valid.", "error")
            return redirect(url_for("upload.dataset", project_id=project_id))
    except Exception as e:
        log.exception("Export dataset gagal: %s", e)
        flash(f"Export gagal: {e}", "error")
        return redirect(url_for("upload.dataset", project_id=project_id))


@bp.route("/<int:project_id>/dataset/<int:image_id>/delete", methods=["POST"])
def delete_image(project_id, image_id):
    img = Image.query.filter_by(id=image_id, project_id=project_id).first_or_404()
    full_path = os.path.join(current_app.config["PROJECTS_FOLDER"], img.filepath)
    if os.path.exists(full_path):
        os.remove(full_path)
    db.session.delete(img)
    db.session.commit()
    log.info("Gambar dihapus | project_id=%s image_id=%s", project_id, image_id)
    flash("Gambar berhasil dihapus.", "success")
    return redirect(url_for("upload.dataset", project_id=project_id))
