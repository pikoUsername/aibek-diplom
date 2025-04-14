import os
from flask import Blueprint, render_template, request, redirect, url_for, current_app

bp = Blueprint("file_manager", __name__, url_prefix="/file_manager")


@bp.route("/", methods=["GET"], endpoint="file_manager")
def file_manager():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if os.path.isfile(os.path.join(current_app.config["UPLOAD_FOLDER"], f))]
    model_files = [f for f in os.listdir(current_app.config["MODEL_FOLDER"]) if os.path.isfile(os.path.join(current_app.config["MODEL_FOLDER"], f))]
    script_files = [f for f in os.listdir(current_app.config["SCRIPT_FOLDER"]) if os.path.isfile(os.path.join(current_app.config["SCRIPT_FOLDER"], f))]
    plot_files = [f for f in os.listdir(current_app.config["PLOT_FOLDER"]) if os.path.isfile(os.path.join(current_app.config["PLOT_FOLDER"], f))]

    return render_template("file_manager.html",
                           data_files=data_files,
                           model_files=model_files,
                           script_files=script_files,
                           plot_files=plot_files)


@bp.route("/delete", methods=["GET"], endpoint="delete_file")
def delete_file():
    file_type = request.args.get("file_type")
    filename = request.args.get("filename")

    folder_map = {
        "data": current_app.config["UPLOAD_FOLDER"],
        "model": current_app.config["MODEL_FOLDER"],
        "script": current_app.config["SCRIPT_FOLDER"],
        "plot": current_app.config["PLOT_FOLDER"]
    }

    folder = folder_map.get(file_type)
    if not folder or not filename:
        return "Қате файл түрі немесе аты", 400

    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return redirect(url_for("file_manager.file_manager"))
    else:
        return f"Файл {filename} табылмады", 404
