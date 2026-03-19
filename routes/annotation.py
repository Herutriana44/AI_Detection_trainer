import os
import json
from flask import Blueprint, request, render_template, jsonify, current_app
from models import db, Project, Image, Annotation

bp = Blueprint("annotation", __name__)

@bp.route("/<int:project_id>/annotate/<int:image_id>")
def annotate(project_id, image_id):
    project = Project.query.get_or_404(project_id)
    image = Image.query.filter_by(id=image_id, project_id=project_id).first_or_404()
    full_path = os.path.join(current_app.config["PROJECTS_FOLDER"], image.filepath)
    if not os.path.exists(full_path):
        return "File not found", 404
    return render_template("annotate.html", project=project, image=image)

@bp.route("/<int:project_id>/annotate/<int:image_id>/classes")
def get_classes(project_id, image_id):
    project = Project.query.get_or_404(project_id)
    classes = set()
    for img in project.images:
        for ann in img.annotations:
            classes.add((ann.class_id, ann.class_name))
    classes = sorted(list(classes), key=lambda x: x[0])
    return jsonify([{"id": c[0], "name": c[1]} for c in classes])

@bp.route("/<int:project_id>/annotate/<int:image_id>/annotations", methods=["GET"])
def get_annotations(project_id, image_id):
    image = Image.query.filter_by(id=image_id, project_id=project_id).first_or_404()
    anns = [
        {
            "id": a.id,
            "class_id": a.class_id,
            "class_name": a.class_name,
            "x_center": a.x_center,
            "y_center": a.y_center,
            "width": a.width,
            "height": a.height
        }
        for a in image.annotations
    ]
    return jsonify(anns)

@bp.route("/<int:project_id>/annotate/<int:image_id>/annotations", methods=["POST"])
def save_annotations(project_id, image_id):
    image = Image.query.filter_by(id=image_id, project_id=project_id).first_or_404()
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Invalid format"}), 400
    
    # Delete existing
    Annotation.query.filter_by(image_id=image_id).delete()
    
    for ann in data:
        obj = Annotation(
            image_id=image_id,
            class_id=ann["class_id"],
            class_name=ann["class_name"],
            x_center=float(ann["x_center"]),
            y_center=float(ann["y_center"]),
            width=float(ann["width"]),
            height=float(ann["height"])
        )
        db.session.add(obj)
    
    image.annotated = len(data) > 0
    db.session.commit()
    return jsonify({"success": True})
