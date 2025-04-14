import math
import os
import pickle
from copyreg import pickle

import pandas as pd
import numpy as np

from flask import Blueprint, current_app, request, render_template
from flask_login import login_required
from sklearn.metrics import mean_squared_error, mean_absolute_error

bp = Blueprint("compare", __name__)


@bp.route("/compare_models", methods=["GET", "POST"])
@login_required
def compare_models():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m for m in os.listdir(current_app.config["MODEL_FOLDER"]) if m.endswith(".pkl")]

    metrics_results = []  # будем сохранять {model_name, MSE, RMSE, MAPE, ...}
    message = None

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        selected_models = request.form.getlist("models")  # если пользователь выбрал несколько
        test_size = float(request.form.get("test_size", "0.2"))

        # Загружаем CSV
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            message = f"Файл {csv_filename} не найден!"
        else:
            df = pd.read_csv(filepath)
            # сортируем и делим на train/test
            df[date_col] = pd.to_datetime(df[date_col])
            df.sort_values(by=date_col, inplace=True)
            ts = df[target_col].values
            split_index = int(len(ts)*(1 - test_size))
            train_data = ts[:split_index]
            test_data = ts[split_index:]

            # Индексируем тест по реальным датам (для графика при желании)
            test_dates = df[date_col].values[split_index:]

            for model_name in selected_models:
                model_path = os.path.join(current_app.config["MODEL_FOLDER"], model_name)
                if not os.path.exists(model_path):
                    continue
                with open(model_path, "rb") as f:
                    model_package = pickle.load(f)
                    model = model_package["model"]

                # Внимание: в SARIMAX нужна целая серия/df. Упрощенно:
                try:
                    forecast_steps = len(test_data)
                    forecast_result = model.get_forecast(steps=forecast_steps)
                    forecast_vals = forecast_result.predicted_mean.values
                except Exception as e:
                    message = f"Ошибка в модели {model_name}: {e}"
                    continue

                # Теперь считаем метрики
                mse = mean_squared_error(test_data, forecast_vals)
                rmse = math.sqrt(mse)
                mae = mean_absolute_error(test_data, forecast_vals)
                mape = np.mean(np.abs((test_data - forecast_vals)/test_data))*100

                metrics_results.append({
                    "model": model_name,
                    "MSE": round(mse, 3),
                    "RMSE": round(rmse, 3),
                    "MAE": round(mae, 3),
                    "MAPE": round(mape, 2)
                })

    return render_template("compare_models.html",
                           data_files=data_files,
                           model_files=model_files,
                           metrics_results=metrics_results,
                           message=message)
