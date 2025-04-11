import os
import pandas as pd
import uuid
import pickle
import matplotlib
matplotlib.use('Agg')  # Для корректной отрисовки в Flask
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, current_app

bp = Blueprint("predict", __name__, url_prefix="/predict")

@bp.route("/", methods=["GET", "POST"])
def predict():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m[:-4] for m in os.listdir(current_app.config["MODEL_FOLDER"])
                   if m.endswith(".pkl") and not (m.startswith("sku_") or m.startswith("exog_"))]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name")
        forecast_steps = int(request.form.get("forecast_steps", 14))
        exog_future_file = request.files.get("exog_future_file")

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")

        if not os.path.exists(filepath) or not os.path.exists(model_path):
            return render_template("predict.html", message="CSV или модель не найдены.", plot_filename=None, data_files=data_files, model_files=model_files)

        df = pd.read_csv(filepath)
        for col in [date_col, target_col]:
            if col not in df.columns:
                return render_template("predict.html", message=f"Столбец {col} отсутствует в данных.", plot_filename=None, data_files=data_files, model_files=model_files)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, target_col])
        df = df.groupby(date_col).agg({target_col: 'sum'}).reset_index()

        # Регулярный временной ряд
        df = df.sort_values(date_col)
        df = df.set_index(date_col).asfreq('D')
        df[target_col] = df[target_col].fillna(method='ffill')
        ts = df[target_col]

        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        model = model_package["model"]
        exog_cols = model_package.get("exog_cols", None)

        if exog_cols:
            if not exog_future_file:
                return render_template("predict.html", message="Нужен CSV с будущими экзогенными переменными.", plot_filename=None, data_files=data_files, model_files=model_files)

            future_exog = pd.read_csv(exog_future_file)
            missing = [col for col in exog_cols if col not in future_exog.columns]
            if missing:
                return render_template("predict.html", message=f"В будущем файле отсутствуют колонки: {missing}", plot_filename=None, data_files=data_files, model_files=model_files)
            future_exog = future_exog[exog_cols]
            if future_exog.shape[0] != forecast_steps:
                return render_template(
                    "predict.html", message="Количество строк в экзогенных данных не совпадает с количеством шагов прогноза.", plot_filename=None, data_files=data_files, model_files=model_files)
        else:
            future_exog = None

        try:
            forecast_result = model.get_forecast(steps=forecast_steps, exog=future_exog)
            forecast_vals = forecast_result.predicted_mean
        except Exception as e:
            return render_template("predict.html", message=f"Ошибка прогнозирования: {e}", plot_filename=None, data_files=data_files, model_files=model_files)

        future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_series = pd.Series(forecast_vals.values, index=future_dates)

        # Визуализация
        plt.figure(figsize=(10, 5))
        plt.plot(ts, label="Исторические данные")
        plt.plot(forecast_series, label="Прогноз", color="orange")
        plt.legend()
        plt.title("Общее предсказание")
        plot_filename = f"predict_{model_name}_{uuid.uuid4().hex[:6]}.png"
        plot_path = os.path.join(current_app.config["PLOT_FOLDER"], plot_filename)
        plt.savefig(plot_path)
        plt.close()

        return render_template("predict.html", message="Прогноз построен!", plot_filename=plot_filename, data_files=data_files, model_files=model_files)

    return render_template("predict.html", message=None, plot_filename=None, data_files=data_files, model_files=model_files)
