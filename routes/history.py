import os
from flask import Blueprint, render_template, current_app

bp = Blueprint("history", __name__, url_prefix="/history")


@bp.route("/", endpoint="history")
def history():
    plots = os.listdir(current_app.config["PLOT_FOLDER"])
    return render_template("history.html", plots=plots)
