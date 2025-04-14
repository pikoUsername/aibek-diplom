import os
import pandas as pd
from flask import Blueprint, request, render_template, current_app

bp = Blueprint("index", __name__)


@bp.route("/", methods=["GET", "POST"], endpoint="index")
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = file.filename
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            df.dropna(inplace=True)
            df.to_csv(filepath, index=False)

            return render_template("index.html", message=f"Файл '{filename}' жүктелді!")
    return render_template("index.html", message=None)
