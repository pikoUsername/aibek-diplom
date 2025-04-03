import os
import pandas as pd
import uuid
import pickle
from flask import Blueprint, render_template, request, current_app
from helpers.script_utils import apply_transform, get_script_list
from statsmodels.tsa.statespace.sarimax import SARIMAX

bp = Blueprint("train", __name__, url_prefix="/train")


@bp.route("/", methods=["GET", "POST"])
def train():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        script_name = request.form.get("script_name")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name", f"model_{uuid.uuid4().hex}")
        # Новое поле: экзогенные переменные (через запятую)
        exog_cols_str = request.form.get("exog_cols", "").strip()

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("train.html", message=f"Файл {csv_filename} не найден!",
                                   data_files=data_files, script_files=script_files)

        df = pd.read_csv(filepath)
        if script_name and script_name != "none":
            script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
            df = apply_transform(df, script_path)

        if date_col not in df.columns or target_col not in df.columns:
            return render_template("train.html", message=f"Нет {date_col} или {target_col} в данных!",
                                   data_files=data_files, script_files=script_files)

        # Если заданы экзогенные колонки, проверяем наличие и извлекаем их
        if exog_cols_str:
            exog_cols = [col.strip() for col in exog_cols_str.split(",") if col.strip()]
            missing_cols = [col for col in exog_cols if col not in df.columns]
            if missing_cols:
                return render_template("train.html", message=f"Колонки {missing_cols} не найдены в данных!",
                                       data_files=data_files, script_files=script_files)
            exog_data = df[exog_cols]
        else:
            exog_cols = None
            exog_data = None

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[target_col]

        # Задаём фиксированные параметры SARIMAX (их можно расширить или сделать вводимыми)
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 7)  # например, недельная сезонность для ежедневных данных

        # Обучаем модель SARIMAX
        try:
            model = SARIMAX(ts, exog=exog_data, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        except Exception as e:
            return render_template("train.html", message=f"Ошибка обучения модели: {e}",
                                   data_files=data_files, script_files=script_files)

        # Сохраняем модель вместе с информацией об экзогенных колонках и параметрах
        model_package = {
            "model": model,
            "exog_cols": exog_cols,
            "order": order,
            "seasonal_order": seasonal_order
        }
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_package, f)

        return render_template("train.html",
                               message=f"Модель '{model_name}' обучена и сохранена!",
                               data_files=data_files, script_files=script_files)

    return render_template("train.html", message=None,
                           data_files=data_files, script_files=script_files)
