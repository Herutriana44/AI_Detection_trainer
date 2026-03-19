import os
from flask import Blueprint, request, redirect, url_for, flash, render_template
from models import db, Project
from utils.logger import get_logger

log = get_logger("projects")
bp = Blueprint("projects", __name__)

@bp.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        if not name:
            flash("Nama project wajib diisi.", "error")
            return render_template("project_form.html", project=None)
        project = Project(name=name, description=description)
        db.session.add(project)
        db.session.commit()
        os.makedirs(os.path.join(request.app.config["PROJECTS_FOLDER"], str(project.id)), exist_ok=True)
        log.info("Project dibuat | id=%s name=%s", project.id, project.name)
        flash("Project berhasil dibuat.", "success")
        return redirect(url_for("dashboard.index"))
    return render_template("project_form.html", project=None)

@bp.route("/<int:project_id>/edit", methods=["GET", "POST"])
def edit(project_id):
    project = Project.query.get_or_404(project_id)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        if not name:
            flash("Nama project wajib diisi.", "error")
            return render_template("project_form.html", project=project)
        project.name = name
        project.description = description
        db.session.commit()
        flash("Project berhasil diperbarui.", "success")
        return redirect(url_for("dashboard.index"))
    return render_template("project_form.html", project=project)

@bp.route("/<int:project_id>/delete", methods=["POST"])
def delete(project_id):
    project = Project.query.get_or_404(project_id)
    project_dir = os.path.join(request.app.config["PROJECTS_FOLDER"], str(project.id))
    if os.path.exists(project_dir):
        import shutil
        shutil.rmtree(project_dir)
    db.session.delete(project)
    db.session.commit()
    log.info("Project dihapus | id=%s", project_id)
    flash("Project berhasil dihapus.", "success")
    return redirect(url_for("dashboard.index"))
