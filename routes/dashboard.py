from flask import Blueprint, render_template
from models import Project

bp = Blueprint("dashboard", __name__)

@bp.route("/")
def index():
    projects = Project.query.order_by(Project.updated_at.desc()).all()
    return render_template("dashboard.html", projects=projects)
