import os
import pandas as pd
import traceback
from flask import Blueprint, render_template, request, current_app, send_file
from helpers.script_utils import get_script_list, apply_transform_code
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

bp = Blueprint("transform", __name__, url_prefix="/transform")

DEFAULT_CODE = '''# Доступные переменные:
# df - DataFrame загруженные данные  
# pd - pandas инструменты редактирования 
# np - numpy математические инструменты 

# Ваш код здесь:
'''


def normalized(code):
    return "".join(code.split())


def get_rows_to_show(code):
    if normalized(code) == normalized(DEFAULT_CODE):
        return 4
    return 4


@bp.route("/", methods=["GET", "POST"])
def transform_data():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])
    message = None
    original_html = None
    transformed_html = None
    script_content = ""
    highlighted_code = ""
    highlight_style = ""
    # При GET-запросе параметр csv_filename берём из GET, а код – дефолтный
    if request.method == "GET":
        selected_csv = request.args.get("csv_filename", "")
        script_code = DEFAULT_CODE
        output_filename = ""
        if selected_csv:
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], selected_csv)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    original_html = df.head(get_rows_to_show(DEFAULT_CODE)).to_html(
                        classes="table table-striped table-bordered", justify="left"
                    )
                except Exception as e:
                    message = f"Ошибка чтения CSV: {e}"
            else:
                message = f"Файл {selected_csv} не найден."
    else:
        # POST-запрос: читаем данные из формы
        selected_csv = request.form.get("csv_filename", "")
        script_code = request.form.get("script_code")
        output_filename = request.form.get("output_filename") or ""
        n = get_rows_to_show(script_code)
        print(script_code)
        if not selected_csv:
            message = "Не выбран CSV файл."
        else:
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], selected_csv)
            if not os.path.exists(filepath):
                message = f"Файл {selected_csv} не найден."
            else:
                try:
                    df = pd.read_csv(filepath)
                    original_html = df.head(n).to_html(
                        classes="table table-striped table-bordered", justify="left"
                    )
                except Exception as e:
                    message = f"Ошибка чтения CSV: {e}"
                    df = None
                if df is not None:
                    try:
                        transformed_df = apply_transform_code(df.copy(), script_code)
                        transformed_html = transformed_df.head(n).to_html(
                            classes="table table-success table-bordered", justify="left"
                        )
                        if output_filename:
                            processed_folder = os.path.join(current_app.config["BASE_DIR"], "data", "processed")
                            os.makedirs(processed_folder, exist_ok=True)
                            output_path = os.path.join(processed_folder, output_filename)
                            transformed_df.to_csv(output_path, index=False)
                            message = f"Данные преобразованы и сохранены как {output_filename}"
                        else:
                            message = "Данные преобразованы. Укажите имя файла, чтобы сохранить результат."
                    except Exception as e:
                        message = f"Ошибка при выполнении скрипта:\n{traceback.format_exc()}"
                    script_content = script_code
                    formatter = HtmlFormatter(style="monokai", noclasses=False)
                    highlighted_code = highlight(script_content, PythonLexer(), formatter)
                    highlight_style = formatter.get_style_defs('.highlight')
    return render_template("transform_data.html",
                           data_files=data_files,
                           script_files=script_files,
                           message=message,
                           original_html=original_html,
                           transformed_html=transformed_html,
                           selected_csv=selected_csv,
                           script_code=script_code,
                           output_filename=output_filename,
                           script_content=script_content,
                           highlighted_code=highlighted_code,
                           highlight_style=highlight_style,
                           get_rows_to_show=get_rows_to_show,
                           default_code=DEFAULT_CODE)


@bp.route("/save_script", methods=["POST"])
def save_script():
    script_code = request.form.get("script_code", "")
    script_filename = request.form.get("script_filename", "script.py")
    save_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_filename)
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(script_code)
        return "Скрипт сохранён успешно.", 200
    except Exception as e:
        return f"Ошибка сохранения скрипта: {str(e)}", 500


@bp.route("/get_script", methods=["GET"])
def get_script():
    script_name = request.args.get("script_name", "")
    if not script_name:
        return "Не указан скрипт", 400
    script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content, 200
    else:
        return "Скрипт не найден", 404


@bp.route("/save_transformed", methods=["POST"])
def save_transformed():
    selected_csv = request.form.get("csv_filename", "")
    script_code = request.form.get("script_code", "")
    output_filename = request.form.get("output_filename", "")
    if not selected_csv or not output_filename:
        return "Не указан CSV файл или имя файла для сохранения.", 400
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], selected_csv)
    if not os.path.exists(filepath):
        return f"Файл {selected_csv} не найден.", 404
    try:
        df = pd.read_csv(filepath)
        transformed_df = apply_transform_code(df.copy(), script_code)
        processed_folder = os.path.join(current_app.config["BASE_DIR"], "data")
        os.makedirs(processed_folder, exist_ok=True)
        output_path = os.path.join(processed_folder, output_filename)
        transformed_df.to_csv(output_path, index=False)
        return f"Данные преобразованы и сохранены как {output_filename}", 200
    except Exception as e:
        return f"Ошибка при сохранении: {e}", 500

@bp.route("/download_transformed", methods=["GET"])
def download_transformed():
    output_filename = request.args.get("output_filename", "")
    if not output_filename:
        return "Имя файла не указано.", 400
    processed_folder = os.path.join(current_app.config["BASE_DIR"], "data")
    output_path = os.path.join(processed_folder, output_filename)
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        return "Файл не найден.", 404
