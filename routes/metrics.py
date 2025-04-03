import os
import pandas as pd
import pickle
from flask import Blueprint, render_template, request, current_app
from helpers.script_utils import apply_transform, get_script_list
from services.metrics_service import calculate_forecast_metrics
from services.model_service import load_model

bp = Blueprint("metrics", __name__, url_prefix="/metrics")

@bp.route("/", methods=["GET", "POST"], endpoint="metrics")
def metrics():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m[:-4] for m in os.listdir(current_app.config["MODEL_FOLDER"]) if m.endswith(".pkl")]
    script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        script_name = request.form.get("script_name")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name")
        test_size = float(request.form.get("test_size", 0.2))

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")

        if not os.path.exists(filepath):
            return render_template("metrics.html", message=f"Файл {csv_filename} не найден!",
                                   metrics_dict=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        if not os.path.exists(model_path):
            return render_template("metrics.html", message=f"Модель {model_name} не найдена!",
                                   metrics_dict=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        df = pd.read_csv(filepath)
        if script_name and script_name != "none":
            script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
            df = apply_transform(df, script_path)

        if date_col not in df.columns or target_col not in df.columns:
            return render_template("metrics.html", message=f"Нет {date_col}/{target_col} в данных!",
                                   metrics_dict=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.dropna(subset=[date_col], inplace=True)
        df.sort_values(by=date_col, inplace=True)

        ts = df.set_index(date_col)[target_col]

        # Загружаем модель и извлекаем информацию об экзогенных переменных
        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        model = model_package["model"]
        exog_cols = model_package.get("exog_cols", None)

        if exog_cols:
            missing_exog = [col for col in exog_cols if col not in df.columns]
            if missing_exog:
                return render_template("metrics.html",
                                       message=f"В данных отсутствуют экзогенные колонки: {missing_exog}",
                                       metrics_dict=None, data_files=data_files,
                                       model_files=model_files, script_files=script_files)
            exog_data = df[exog_cols]
        else:
            exog_data = None

        n = len(ts)
        split_idx = int(n * (1 - test_size))
        train_data = ts.iloc[:split_idx]
        test_data = ts.iloc[split_idx:]

        if exog_data is not None:
            train_exog = exog_data.iloc[:split_idx]
            test_exog = exog_data.iloc[split_idx:]
        else:
            train_exog = None
            test_exog = None

        # Выполняем прогноз для тестового набора с использованием SARIMAX
        try:
            forecast_result = model.get_forecast(steps=len(test_data), exog=test_exog)
            forecast_vals = forecast_result.predicted_mean
        except Exception as e:
            return render_template("metrics.html",
                                   message=f"Ошибка при прогнозе: {e}",
                                   metrics_dict=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        forecast_series = pd.Series(forecast_vals, index=test_data.index)

        common_index = test_data.index.intersection(forecast_series.index)
        test_data = test_data.loc[common_index].dropna()
        forecast_series = forecast_series.loc[common_index].dropna()

        if len(test_data) == 0 or len(forecast_series) == 0:
            return render_template("metrics.html",
                                   message="Недостаточно данных для метрик.",
                                   metrics_dict=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        metrics_dict = calculate_forecast_metrics(test_data, forecast_series)

        return render_template("metrics.html", message="Метрики успешно рассчитаны!",
                               metrics_dict=metrics_dict, data_files=data_files,
                               model_files=model_files, script_files=script_files)

    return render_template("metrics.html", message=None, metrics_dict=None,
                           data_files=data_files, model_files=model_files,
                           script_files=script_files)
