import os
import numpy as np
import pandas as pd
import pickle
from flask import Blueprint, render_template, request, current_app, send_file
from helpers.script_utils import apply_transform, get_script_list
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

bp = Blueprint("metrics", __name__, url_prefix="")

@bp.route("/metrics", methods=["GET", "POST"], endpoint="metrics")
def metrics():

    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m[:-4] for m in os.listdir(current_app.config["MODEL_FOLDER"]) if m.endswith(".pkl")]
    script_files = get_script_list("scripts/")  # предполагается, что папка с пользовательскими скриптами хранится здесь

    if request.method == "POST":

        csv_filename = request.form.get("csv_filename")
        script_name = request.form.get("script_name", "none")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name")
        test_size = float(request.form.get("test_size", 0.2))


        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("metrics.html",
                                   message=f"Файл {csv_filename} не найден!",
                                   metrics_dict=None,
                                   data_files=data_files,
                                   model_files=model_files,
                                   script_files=script_files)


        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        if not os.path.exists(model_path):
            return render_template("metrics.html",
                                   message=f"Модель {model_name} не найдена!",
                                   metrics_dict=None,
                                   data_files=data_files,
                                   model_files=model_files,
                                   script_files=script_files)


        df = pd.read_csv(filepath)


        if script_name != "none":
            script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
            df = apply_transform(df, script_path)


        if date_col not in df.columns or target_col not in df.columns:
            return render_template("metrics.html",
                                   message=f"Нет колонок {date_col} или {target_col} в данных!",
                                   metrics_dict=None,
                                   data_files=data_files,
                                   model_files=model_files,
                                   script_files=script_files)


        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.sort_values(by=date_col, inplace=True)
        df.dropna(subset=[date_col], inplace=True)


        ts = df.set_index(date_col)[target_col].astype(float)
        ts.dropna(inplace=True)


        n = len(ts)
        split_idx = int(n * (1 - test_size))
        train_data = ts.iloc[:split_idx]
        test_data = ts.iloc[split_idx:]

        if len(train_data) == 0 or len(test_data) == 0:

            metrics_dict = {"MSE": 0.0, "RMSE": 0.0, "MAPE (%)": 0.0}
            return render_template("metrics.html",
                                   message="Недостаточно данных, выданы фиктивные метрики.",
                                   metrics_dict=metrics_dict,
                                   data_files=data_files,
                                   model_files=model_files,
                                   script_files=script_files)


        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        if isinstance(loaded_model, dict) and "model" in loaded_model:
            base_model = loaded_model["model"]
            exog_cols = loaded_model.get("exog_cols", None)
        else:
            base_model = loaded_model
            exog_cols = None

        if exog_cols:
            missing_exog = [col for col in exog_cols if col not in df.columns]
            if missing_exog:
                return render_template("metrics.html",
                                       message=f"Отсутствуют экзогенные переменные: {missing_exog}",
                                       metrics_dict=None,
                                       data_files=data_files,
                                       model_files=model_files,
                                       script_files=script_files)
            exog_data = df[exog_cols]
            train_exog = exog_data.iloc[:split_idx]
            test_exog = exog_data.iloc[split_idx:]
        else:
            train_exog = None
            test_exog = None


        if hasattr(base_model, "fit") and not isinstance(base_model, SARIMAXResults):
            try:
                if train_exog is not None:
                    fitted_model = base_model.fit(train_data, exog=train_exog)
                else:
                    fitted_model = base_model.fit(train_data)
            except Exception as e:
                return render_template("metrics.html",
                                       message=f"Ошибка переобучения модели: {e}",
                                       metrics_dict=None,
                                       data_files=data_files,
                                       model_files=model_files,
                                       script_files=script_files)
        else:
            fitted_model = base_model


        try:
            if test_exog is not None:

                if hasattr(fitted_model, "get_forecast"):
                    forecast_result = fitted_model.get_forecast(steps=len(test_data), exog=test_exog)
                    forecast_vals = forecast_result.predicted_mean
                else:
                    forecast_vals = fitted_model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, exog=test_exog)
            else:

                if hasattr(fitted_model, "get_forecast"):
                    forecast_result = fitted_model.get_forecast(steps=len(test_data))
                    forecast_vals = forecast_result.predicted_mean
                else:
                    forecast_vals = fitted_model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
        except Exception as e:
            return render_template("metrics.html",
                                   message=f"Ошибка при прогнозировании: {e}",
                                   metrics_dict=None,
                                   data_files=data_files,
                                   model_files=model_files,
                                   script_files=script_files)


        test_array = test_data.values
        forecast_vals = np.array(forecast_vals)
        min_len = min(len(test_array), len(forecast_vals))
        test_array = test_array[:min_len]
        forecast_vals = forecast_vals[:min_len]


        if min_len == 0:
            fallback_len = len(test_data)
            mean_train = train_data.mean()
            forecast_vals = np.array([mean_train] * fallback_len)
            test_array = test_data.values
            min_len = len(test_array)
            if min_len == 0:
                metrics_dict = {"MSE": 0.0, "RMSE": 0.0, "MAPE (%)": 0.0}
                return render_template("metrics.html",
                                       message="Нет данных для прогноза, выданы фиктивные метрики.",
                                       metrics_dict=metrics_dict,
                                       data_files=data_files,
                                       model_files=model_files,
                                       script_files=script_files)


        test_array = np.nan_to_num(test_array, nan=0.0)
        forecast_vals = np.nan_to_num(forecast_vals, nan=0.0)


        mse_value = mean_squared_error(test_array, forecast_vals)
        rmse_value = np.sqrt(mse_value)
        eps = 1e-8
        mape_array = np.abs((test_array - forecast_vals) / (test_array + eps)) * 100
        mape_value = np.mean(mape_array)

        metrics_dict = {
            "MSE": round(mse_value, 3),
            "RMSE": round(rmse_value, 3),
            "MAPE (%)": round(mape_value, 3)
        }

        return render_template("metrics.html",
                               message="Метрики успешно рассчитаны!",
                               metrics_dict=metrics_dict,
                               data_files=data_files,
                               model_files=model_files,
                               script_files=script_files)


    return render_template("metrics.html",
                           message=None,
                           metrics_dict=None,
                           data_files=data_files,
                           model_files=model_files,
                           script_files=script_files)
