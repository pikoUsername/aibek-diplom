import os
import pandas as pd
import uuid
import pickle
import json
from flask import Blueprint, render_template, request, current_app
from helpers.script_utils import apply_transform, get_script_list
from statsmodels.tsa.statespace.sarimax import SARIMAX

bp = Blueprint("train", __name__, url_prefix="/train")

@bp.route("/", methods=["GET", "POST"], endpoint="train")
def train():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        script_name = request.form.get("script_name")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name", f"model_{uuid.uuid4().hex}")
        exog_cols_str = request.form.get("exog_cols", "").strip()

        # Считываем динамические параметры модели
        order_str = request.form.get("order", "").strip()           # ожидается "1,1,1"
        seasonal_order_str = request.form.get("seasonal_order", "").strip()  # ожидается "1,1,1,7"
        additional_params_str = request.form.get("additional_params", "{}").strip()

        # Парсим order
        if order_str:
            try:
                order = tuple(int(x.strip()) for x in order_str.split(","))
                if len(order) != 3:
                    raise ValueError
            except Exception:
                return render_template("train.html",
                                       message="Неверный формат параметра order. Ожидается: p,d,q",
                                       data_files=data_files, script_files=script_files)
        else:
            order = (1, 1, 1)

        # Парсим seasonal_order
        if seasonal_order_str:
            try:
                seasonal_order = tuple(int(x.strip()) for x in seasonal_order_str.split(","))
                if len(seasonal_order) != 4:
                    raise ValueError
            except Exception:
                return render_template("train.html",
                                       message="Неверный формат параметра seasonal_order. Ожидается: P,D,Q,m",
                                       data_files=data_files, script_files=script_files)
        else:
            seasonal_order = (1, 1, 1, 7)

        # Парсим дополнительные параметры из JSON
        try:
            additional_params = json.loads(additional_params_str)
        except Exception as e:
            return render_template("train.html",
                                   message=f"Неверный формат дополнительных параметров: {e}",
                                   data_files=data_files, script_files=script_files)

        # Обработка экзогенных переменных, если заданы
        if exog_cols_str:
            exog_cols = [col.strip() for col in exog_cols_str.split(",") if col.strip()]
        else:
            exog_cols = None

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("train.html",
                                   message=f"Файл {csv_filename} не найден!",
                                   data_files=data_files, script_files=script_files)

        df = pd.read_csv(filepath)
        if script_name and script_name != "none":
            script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
            df = apply_transform(df, script_path)

        if date_col not in df.columns or target_col not in df.columns:
            return render_template("train.html",
                                   message=f"Нет {date_col} или {target_col} в данных!",
                                   data_files=data_files, script_files=script_files)

        if exog_cols:
            missing_cols = [col for col in exog_cols if col not in df.columns]
            if missing_cols:
                return render_template("train.html",
                                       message=f"Колонки {missing_cols} не найдены в данных!",
                                       data_files=data_files, script_files=script_files)
            exog_data = df[exog_cols]
        else:
            exog_data = None

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[target_col]

        # Обучаем модель SARIMAX с динамическими параметрами
        try:
            model = SARIMAX(ts, exog=exog_data, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                            **additional_params).fit(disp=False)
        except Exception as e:
            return render_template("train.html",
                                   message=f"Ошибка обучения модели: {e}",
                                   data_files=data_files, script_files=script_files)

        # Сохраняем модель вместе с информацией о параметрах и экзогенных колонках
        model_package = {
            "model": model,
            "exog_cols": exog_cols,
            "order": order,
            "seasonal_order": seasonal_order,
            "additional_params": additional_params
        }
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_package, f)

        return render_template("train.html",
                               message=f"Модель '{model_name}' обучена и сохранена!",
                               data_files=data_files, script_files=script_files)

    return render_template("train.html", message=None,
                           data_files=data_files, script_files=script_files)
