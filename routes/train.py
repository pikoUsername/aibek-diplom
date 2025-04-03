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
    # Получаем список CSV-файлов
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name", f"model_{uuid.uuid4().hex}")
        # Динамические параметры модели
        order_str = request.form.get("order", "").strip()           # ожидается "p,d,q"
        seasonal_order_str = request.form.get("seasonal_order", "").strip()  # ожидается "P,D,Q,m"
        additional_params_str = request.form.get("additional_params", "{}").strip()
        # Опционально: список экзогенных колонок (если модель планируется с экзогенными)
        exog_cols_str = request.form.get("exog_cols", "").strip()

        # Парсинг order
        try:
            order = tuple(int(x.strip()) for x in order_str.split(",")) if order_str else (1, 1, 1)
            if len(order) != 3:
                raise ValueError
        except Exception:
            return render_template("train.html", message="Неверный формат order (ожидается: p,d,q)",
                                   data_files=data_files)
        # Парсинг seasonal_order
        try:
            seasonal_order = tuple(int(x.strip()) for x in seasonal_order_str.split(",")) if seasonal_order_str else (1, 1, 1, 7)
            if len(seasonal_order) != 4:
                raise ValueError
        except Exception:
            return render_template("train.html", message="Неверный формат seasonal_order (ожидается: P,D,Q,m)",
                                   data_files=data_files)
        # Парсинг дополнительных параметров
        try:
            additional_params = json.loads(additional_params_str)
        except Exception as e:
            return render_template("train.html", message=f"Неверный формат дополнительных параметров: {e}",
                                   data_files=data_files)
        # Обработка экзогенных колонок (если указаны)
        if exog_cols_str:
            exog_cols = [col.strip() for col in exog_cols_str.split(",") if col.strip()]
        else:
            exog_cols = None

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("train.html", message=f"Файл {csv_filename} не найден!", data_files=data_files)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return render_template("train.html", message=f"Ошибка чтения CSV: {e}", data_files=data_files)

        # Проверка необходимых столбцов
        for col in [date_col, target_col]:
            if col not in df.columns:
                return render_template("train.html", message=f"Столбец {col} отсутствует в данных!", data_files=data_files)

        # Если заданы экзогенные переменные, проверяем их наличие
        if exog_cols:
            missing = [col for col in exog_cols if col not in df.columns]
            if missing:
                return render_template("train.html", message=f"Экзогенные столбцы отсутствуют: {missing}", data_files=data_files)
            exog_data = df[exog_cols]
        else:
            exog_data = None

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[target_col]

        try:
            model = SARIMAX(ts, exog=exog_data, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                            **additional_params).fit(disp=False)
        except Exception as e:
            return render_template("train.html", message=f"Ошибка обучения модели: {e}", data_files=data_files)

        # Сохраняем модель (рекомендуется добавить префикс для общего типа, например, "model_")
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

        return render_template("train.html", message=f"Модель '{model_name}' обучена и сохранена!", data_files=data_files)

    return render_template("train.html", message=None, data_files=data_files)
