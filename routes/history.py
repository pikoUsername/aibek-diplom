import os
from flask import Blueprint, render_template, current_app, session
from flask_login import login_required, current_user

from db.models import db, Plot
from services.login_manager import login_manager

bp = Blueprint("history", __name__, url_prefix="/history")


@bp.route("/", endpoint="history")
@login_required
def history():
    plots: list[Plot] = db.session.query(Plot).where(Plot.user_id == current_user.id).all()

    plots_paths = [plot.plot_path for plot in plots]

    return render_template("history.html", plots=plots_paths)
