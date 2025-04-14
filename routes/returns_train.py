import os
import pandas as pd
import uuid
import pickle
import json
from flask import Blueprint, render_template, request, current_app
from flask_login import login_required
from statsmodels.tsa.statespace.sarimax import SARIMAX

bp = Blueprint("returns_train", __name__, url_prefix="/train_returns")

@bp.route("/", methods=["GET", "POST"])
@login_required
def returns_train():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        returns_col = request.form.get("returns_col")
        model_name = request.form.get("model_name", f"returns_{uuid.uuid4().hex}")
        order_str = request.form.get("order", "").strip()
        seasonal_order_str = request.form.get("seasonal_order", "").strip()
        additional_params_str = request.form.get("additional_params", "{}").strip()

        try:
            order = tuple(int(x.strip()) for x in order_str.split(",")) if order_str else (1,1,1)
            if len(order) != 3: raise ValueError
        except Exception:
            return render_template("returns_train.html", message="Неверный формат order.", data_files=data_files)
        try:
            seasonal_order = tuple(int(x.strip()) for x in seasonal_order_str.split(",")) if seasonal_order_str else (1,1,1,7)
            if len(seasonal_order) != 4: raise ValueError
        except Exception:
            return render_template("returns_train.html", message="Неверный формат seasonal_order.", data_files=data_files)
        try:
            additional_params = json.loads(additional_params_str)
        except Exception as e:
            return render_template("returns_train.html", message=f"Неверный формат дополнительных параметров: {e}", data_files=data_files)

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("returns_train.html", message=f"Файл {csv_filename} не найден!", data_files=data_files)
        df = pd.read_csv(filepath)
        for col in [date_col, returns_col]:
            if col not in df.columns:
                return render_template("returns_train.html", message=f"Столбец {col} отсутствует.", data_files=data_files)
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[returns_col]

        try:
            model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                            **additional_params).fit(disp=False)
        except Exception as e:
            return render_template("returns_train.html", message=f"Ошибка обучения модели: {e}", data_files=data_files)

        model_package = {
            "model": model,
            "order": order,
            "seasonal_order": seasonal_order,
            "additional_params": additional_params
        }
        save_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model_package, f)

        return render_template("returns_train.html", message=f"Модель '{model_name}' обучена и сохранена!", data_files=data_files)
    return render_template("returns_train.html", message=None, data_files=data_files)
