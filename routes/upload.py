import os
import uuid
from flask import Blueprint, request, redirect, url_for, flash, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
from models import db, Project, Image

bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/<int:project_id>/image/<path:filepath>")
def serve_image(project_id, filepath):
    Project.query.get_or_404(project_id)
    base = request.app.config["PROJECTS_FOLDER"]
    return send_from_directory(base, filepath)

@bp.route("/<int:project_id>/dataset")
def dataset(project_id):
    project = Project.query.get_or_404(project_id)
    images = project.images
    return render_template("dataset.html", project=project, images=images)

@bp.route("/<int:project_id>/dataset/upload", methods=["POST"])
def upload_files(project_id):
    project = Project.query.get_or_404(project_id)
    project_dir = os.path.join(request.app.config["PROJECTS_FOLDER"], str(project_id), "images")
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
                filepath=os.path.relpath(filepath, request.app.config["PROJECTS_FOLDER"]),
                width=w,
                height=h
            )
            db.session.add(image)
            count += 1
    
    db.session.commit()
    flash(f"{count} gambar berhasil diupload.", "success")
    return redirect(url_for("upload.dataset", project_id=project_id))

@bp.route("/<int:project_id>/dataset/<int:image_id>/delete", methods=["POST"])
def delete_image(project_id, image_id):
    img = Image.query.filter_by(id=image_id, project_id=project_id).first_or_404()
    full_path = os.path.join(request.app.config["PROJECTS_FOLDER"], img.filepath)
    if os.path.exists(full_path):
        os.remove(full_path)
    db.session.delete(img)
    db.session.commit()
    flash("Gambar berhasil dihapus.", "success")
    return redirect(url_for("upload.dataset", project_id=project_id))
