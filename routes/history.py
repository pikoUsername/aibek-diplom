from pathlib import Path

from flask import Blueprint, render_template
from flask_login import login_required, current_user

from db.models import db, Plot

bp = Blueprint("history", __name__, url_prefix="/history")


@bp.route("/", endpoint="history")
@login_required
def history():
    plots: list[Plot] = db.session.query(Plot).where(Plot.user_id == int(current_user.id)).all()

    plots_paths = [plot.plot_filename for plot in plots]

    return render_template("history.html", plots=plots_paths)
