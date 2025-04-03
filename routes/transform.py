import os
import traceback

import pandas as pd
from flask import Blueprint, render_template, request, current_app, Response
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from helpers.script_utils import get_script_list

bp = Blueprint("transform", __name__, url_prefix="/transform")

# Дефолтный код, если пользователь не ввёл свой код
DEFAULT_CODE = '''# Доступные переменные:
# df - DataFrame загруженные данные  
# pd - pandas инструменты редактирования 
# np - numpy математические инструменты 

# Ваш код здесь:
'''


def normalized(code):
    return "".join(code.split())


def get_rows_to_show(code):
    # Если код пустой или равен дефолтному (без пробелов) – показать 8 строк, иначе 4
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
    script_content = None
    highlighted_code = None
    highlight_style = None
    selected_csv = request.args.get("csv_filename", "")
    output_filename = ""
    script_code = ""

    # GET-запрос: если выбран CSV, показываем исходные данные
    if request.method == "GET" and selected_csv:
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

    # POST-запрос: выполняем скрипт
    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        script_code = request.form.get("script_code") or DEFAULT_CODE
        output_filename = request.form.get("output_filename") or ""
        selected_csv = csv_filename
        n = get_rows_to_show(script_code)

        if not csv_filename:
            message = "Не выбран CSV файл."
        elif not script_code:
            message = "Введите скрипт для преобразования данных."
        else:
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
            if not os.path.exists(filepath):
                message = f"Файл {csv_filename} не найден."
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
                        local_env = {"df": df.copy()}
                        global_env = {"__builtins__": __builtins__, "pd": pd, "np": __import__("numpy")}
                        exec(script_code, global_env, local_env)
                        transformed_df = local_env.get("df", df)
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

                        script_content = script_code
                        formatter = HtmlFormatter(style="monokai", noclasses=False)
                        highlighted_code = highlight(script_content, PythonLexer(), formatter)
                        highlight_style = formatter.get_style_defs('.highlight')
                    except Exception as e:
                        message = f"Ошибка при выполнении скрипта:\n{traceback.format_exc()}"
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


@bp.route("/get_script", methods=["GET"])
def get_script():
    script_name = request.args.get("script_name", "")
    if not script_name:
        return Response("Не указан скрипт", status=400)
    script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content, mimetype="text/plain")
    else:
        return Response("Скрипт не найден", status=404)


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
