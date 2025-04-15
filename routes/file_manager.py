import os
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, current_app, send_file
from flask_login import login_required
from werkzeug.utils import secure_filename

bp = Blueprint("file_manager", __name__, url_prefix="/file_manager")



@bp.route("/", methods=["GET"], endpoint="file_manager")
@login_required
def file_manager():
    # Получение поиска, если передан параметр query
    query = request.args.get("query", "").lower()

    def filter_files(file_list):
        if query:
            return [f for f in file_list if query in f.lower()]
        return file_list

    data_files_all = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"])
                      if os.path.isfile(os.path.join(current_app.config["UPLOAD_FOLDER"], f))]
    model_files_all = [f for f in os.listdir(current_app.config["MODEL_FOLDER"])
                       if os.path.isfile(os.path.join(current_app.config["MODEL_FOLDER"], f))]
    script_files_all = [f for f in os.listdir(current_app.config["SCRIPT_FOLDER"])
                        if os.path.isfile(os.path.join(current_app.config["SCRIPT_FOLDER"], f))]
    plot_files_all = [f for f in os.listdir(current_app.config["PLOT_FOLDER"])
                      if os.path.isfile(os.path.join(current_app.config["PLOT_FOLDER"], f))]

    data_files = filter_files(data_files_all)
    model_files = filter_files(model_files_all)
    script_files = filter_files(script_files_all)
    plot_files = filter_files(plot_files_all)

    return render_template("file_manager.html",
                           data_files=data_files,
                           model_files=model_files,
                           script_files=script_files,
                           plot_files=plot_files)


@bp.route("/delete", methods=["GET"], endpoint="delete_file")
@login_required
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

    filepath = os.path.join(folder, secure_filename(filename))
    if os.path.exists(filepath):
        os.remove(filepath)
        return redirect(url_for("file_manager.file_manager"))
    else:
        return f"Файл {filename} табылмады", 404


@bp.route("/rename", methods=["POST"], endpoint="rename_file")
@login_required
def rename_file():
    file_type = request.form.get("file_type")
    old_filename = request.form.get("old_filename")
    new_filename = request.form.get("new_filename")
    if not file_type or not old_filename or not new_filename:
        return "Қате: керекті ақпарат жеткіліксіз", 400
    folder_map = {
        "data": current_app.config["UPLOAD_FOLDER"],
        "model": current_app.config["MODEL_FOLDER"],
        "script": current_app.config["SCRIPT_FOLDER"],
        "plot": current_app.config["PLOT_FOLDER"]
    }
    folder = folder_map.get(file_type)
    old_path = os.path.join(folder, secure_filename(old_filename))
    new_path = os.path.join(folder, secure_filename(new_filename))
    if not os.path.exists(old_path):
        return f"Файл {old_filename} табылмады", 404
    if os.path.exists(new_path):
        return f"Файл {new_filename} алдын ала бар", 400
    try:
        os.rename(old_path, new_path)
        return f"Файл атауы {old_filename} -ден {new_filename} -ге өзгертілді"
    except Exception as e:
        return f"Переименование қатесі: {e}", 500


@bp.route("/view_file", methods=["GET"], endpoint="view_file")
@login_required
def view_file():
    file_type = request.args.get("file_type")
    filename = request.args.get("filename")
    folder_map = {
        "data": current_app.config["UPLOAD_FOLDER"],
        "model": current_app.config["MODEL_FOLDER"],
        "script": current_app.config["SCRIPT_FOLDER"],
        "plot": current_app.config["PLOT_FOLDER"]
    }
    folder = folder_map.get(file_type)
    filename = secure_filename(filename)
    if not folder or not filename:
        return "Қате: файл түрі немесе аты көрсетілмеген", 400
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        return f"Файл {filename} табылмады", 404

    ext = os.path.splitext(filename)[1].lower()
    if file_type == "plot":
        return send_file(filepath)
    if file_type == "model":
        # Модель файлдарын редактировать болмайды.
        return render_template("view_binary.html", filename=filename)
    if file_type == "data":
        if ext == ".csv":
            try:
                df = pd.read_csv(filepath)
                table_html = df.to_html(classes="table table-striped", index=False)
                return render_template("view_table.html", filename=filename, table_html=table_html)
            except Exception as e:
                return f"CSV файлын оқу қатесі: {e}", 500
        elif ext in [".xls", ".xlsx"]:
            try:
                df = pd.read_excel(filepath)
                table_html = df.to_html(classes="table table-striped", index=False)
                return render_template("view_table.html", filename=filename, table_html=table_html)
            except Exception as e:
                return f"Excel файлын оқу қатесі: {e}", 500
        else:
            # Если не CSV/Excel – откроем как текстовый файл через CodeMirror
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return render_template("view_file.html", filename=filename, content=content, file_type=file_type, mode="null")
    if file_type == "script":
        mode = "python" if ext == ".py" else "null"
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template("view_file.html", filename=filename, content=content, file_type=file_type, mode=mode)
    # Фолбэк
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return render_template("view_file.html", filename=filename, content=content, file_type=file_type, mode="null")


@bp.route("/save_file", methods=["POST"], endpoint="save_file")
@login_required
def save_file():
    file_type = request.form.get("file_type")
    filename = request.form.get("filename")
    content = request.form.get("content")
    if file_type not in ["data", "script"]:
        return "Бұл файлды өңдеуге тыйым салынған", 403
    folder_map = {
        "data": current_app.config["UPLOAD_FOLDER"],
        "script": current_app.config["SCRIPT_FOLDER"]
    }
    folder = folder_map.get(file_type)
    if not folder or not filename:
        return "Қате: файл түрі немесе аты көрсетілмеген", 400
    filepath = os.path.join(folder, secure_filename(filename))
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return "Файл сәтті сақталды", 200
    except Exception as e:
        return f"Файлды сақтау қатесі: {e}", 500
