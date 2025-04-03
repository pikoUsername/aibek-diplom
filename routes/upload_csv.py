import os
import uuid
from flask import Blueprint, request, render_template, current_app

bp = Blueprint("upload", __name__, url_prefix="/upload_script")


@bp.route("/", methods=["GET", "POST"], endpoint="upload_script")
def upload_script():
    if request.method == "POST":
        script_code = request.form.get("script_code")
        script_name = request.form.get("script_name")

        if not script_name:
            script_name = f"script_{uuid.uuid4().hex}"
        if not script_name.endswith(".py"):
            script_name += ".py"

        script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        return render_template("upload_script.html", message=f"Скрипт '{script_name}' сохранён!")

    return render_template("upload_script.html", message=None)
