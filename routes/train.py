import os
import pandas as pd
import uuid
import pickle
import json
from flask import Blueprint, render_template, request, current_app
from statsmodels.tsa.statespace.sarimax import SARIMAX

bp = Blueprint("train", __name__, url_prefix="/train")


@bp.route("/", methods=["GET", "POST"])
def train():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name", f"model_{uuid.uuid4().hex}")
        order_str = request.form.get("order", "1,1,1").strip()
        seasonal_order_str = request.form.get("seasonal_order", "1,1,1,7").strip()
        additional_params_str = request.form.get("additional_params", "{}").strip()
        exog_cols_str = request.form.get("exog_cols", "").strip()

        try:
            if order_str == "":
                order_str = "1,1,1"
            order = tuple(map(int, order_str.split(",")))
            if seasonal_order_str == "":
                seasonal_order_str = "1,1,1,7"
            seasonal_order = tuple(map(int, seasonal_order_str.split(",")))
            additional_params = json.loads(additional_params_str)
        except Exception as e:
            return render_template("train.html", message=f"Ошибка параметров модели: {e}", data_files=data_files)

        exog_cols = [col.strip() for col in exog_cols_str.split(",")] if exog_cols_str else None

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("train.html", message="Файл не найден", data_files=data_files)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return render_template("train.html", message=f"Ошибка чтения CSV: {e}", data_files=data_files)

        # Проверка необходимых столбцов
        for col in [date_col, target_col]:
            if col not in df.columns:
                return render_template("train.html", message=f"Столбец {col} отсутствует в данных!", data_files=data_files)

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, target_col])
        df = df.groupby(date_col).agg({target_col: 'sum'}).reset_index()

        # Регулярный временной ряд
        df = df.sort_values(date_col)
        df = df.set_index(date_col).asfreq('D')
        df[target_col] = df[target_col].fillna(method='ffill')

        exog_data = df[exog_cols] if exog_cols else None
        if exog_data is None and exog_cols is not None:
            return render_template("train.html", message=f"Экзогенные столбцы отсутствуют", data_files=data_files)

        try:
            model = SARIMAX(df[target_col], exog=exog_data, order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            **additional_params).fit(disp=False)
        except Exception as e:
            return render_template("train.html", message=f"Ошибка обучения модели: {e}", data_files=data_files)

        model_package = {
            "model": model,
            "order": order,
            "seasonal_order": seasonal_order,
            "additional_params": additional_params,
            "exog_cols": exog_cols
        }

        save_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model_package, f)

        return render_template("train.html", message=f"Модель '{model_name}' успешно обучена!", data_files=data_files)

    return render_template("train.html", message=None, data_files=data_files)
