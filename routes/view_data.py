import os
import pandas as pd
from flask import Blueprint, render_template, request, current_app
from helpers.script_utils import get_script_list, apply_transform
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

bp = Blueprint("view_data", __name__, url_prefix="/view_data")

@bp.route("/", methods=["GET", "POST"])
def view_data():
    # Получаем список CSV-файлов из папки данных
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    # Получаем список скриптов из папки скриптов (передаем путь)
    script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        script_name = request.form.get("script_name", "none")

        raw_html = None
        transformed_html = None
        script_content = None
        highlighted_code = None
        highlight_style = None
        message = ""

        try:
            if csv_filename and csv_filename in data_files:
                # Читаем CSV
                filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
                raw_df = pd.read_csv(filepath)

                # Преобразуем первые 20 строк в HTML-таблицу
                raw_html = raw_df.head(20).to_html(
                    classes="table table-striped table-bordered",
                    justify="left"
                )
                message = f"Показаны данные из файла {csv_filename}"

                # Если выбран скрипт, применяем его к данным
                if script_name != "none" and script_name in script_files:
                    script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
                    transformed_df = apply_transform(raw_df.copy(), script_path)
                    transformed_html = transformed_df.head(20).to_html(
                        classes="table table-success table-bordered",
                        justify="left"
                    )

                    with open(script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()

                    formatter = HtmlFormatter(style="monokai", noclasses=False)
                    highlighted_code = highlight(script_content, PythonLexer(), formatter)
                    highlight_style = formatter.get_style_defs('.highlight')

                    message += f" с применением скрипта {script_name}"
            else:
                message = "Неверное имя CSV или файл не выбран!"
        except Exception as e:
            current_app.logger.exception("Ошибка при обработке view_data: %s", e)
            # Если произошла ошибка, сохраняем сообщение и обнуляем результаты для отображения
            message = f"Произошла ошибка: {e}"
            raw_html = None
            transformed_html = None
            script_content = None
            highlighted_code = None
            highlight_style = None

        return render_template(
            "view_data.html",
            data_files=data_files,
            script_files=script_files,
            message=message,
            raw_html=raw_html,
            transformed_html=transformed_html,
            selected_csv=csv_filename,
            selected_script=script_name,
            script_content=script_content,
            highlighted_code=highlighted_code,
            highlight_style=highlight_style
        )
    else:
        return render_template(
            "view_data.html",
            data_files=data_files,
            script_files=script_files,
            message=None,
            raw_html=None,
            transformed_html=None,
            selected_csv=None,
            selected_script=None,
            script_content=None,
            highlighted_code=None,
            highlight_style=None
        )
