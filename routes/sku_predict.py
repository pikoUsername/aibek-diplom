import os
import pandas as pd
import uuid
import pickle
from flask import Blueprint, render_template, request, current_app
import matplotlib.pyplot as plt
from flask_login import login_required, current_user

from db.models import Plot, db

bp = Blueprint("sku_predict", __name__, url_prefix="/predict_sku")

@bp.route("/", methods=["GET", "POST"])
@login_required
def sku_predict():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m[:-4] for m in os.listdir(current_app.config["MODEL_FOLDER"]) if m.endswith(".pkl") and m.startswith("sku_")]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        sku_col = request.form.get("sku_col")
        sales_col = request.form.get("sales_col")
        sku_value = request.form.get("sku_value")
        model_name = request.form.get("model_name")
        forecast_steps = int(request.form.get("forecast_steps", 14))

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        if not os.path.exists(filepath) or not os.path.exists(model_path):
            return render_template("sku_predict.html", message="CSV или модель не найдены.", plot_filename=None, data_files=data_files, model_files=model_files)

        df = pd.read_csv(filepath)
        for col in [date_col, sku_col, sales_col]:
            if col not in df.columns:
                return render_template("sku_predict.html", message=f"Столбец {col} отсутствует.", plot_filename=None, data_files=data_files, model_files=model_files)
        sku_df = df[df[sku_col] == sku_value]
        if sku_df.empty:
            return render_template("sku_predict.html", message=f"Нет данных для SKU: {sku_value}", plot_filename=None, data_files=data_files, model_files=model_files)
        sku_df[date_col] = pd.to_datetime(sku_df[date_col])
        sku_df.sort_values(by=date_col, inplace=True)
        ts = sku_df.set_index(date_col)[sales_col]

        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        model = model_package["model"]

        try:
            forecast = model.get_forecast(steps=forecast_steps).predicted_mean
        except Exception as e:
            return render_template("sku_predict.html", message=f"Ошибка прогнозирования: {e}", plot_filename=None, data_files=data_files, model_files=model_files)

        future_dates = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='D')[1:]
        forecast_series = pd.Series(forecast, index=future_dates)

        plt.figure(figsize=(10, 5))
        plt.plot(ts, label="Исторические данные")
        plt.plot(forecast_series, label="Прогноз")
        plt.legend()
        plt.title(f"Прогноз для SKU: {sku_value}")
        plot_filename = f"sku_forecast_{sku_value}_{uuid.uuid4().hex[:6]}.png"
        plot_path = os.path.join(current_app.config["PLOT_FOLDER"], plot_filename)
        plt.savefig(plot_path)
        plt.close()

        plot = Plot(user_id=current_user.id, plot_filename=plot_path)

        db.session.add(plot)
        db.session.commit()

        return render_template("sku_predict.html", message="Прогноз сформирован!", plot_filename=plot_filename, data_files=data_files, model_files=model_files)
    return render_template("sku_predict.html", message=None, plot_filename=None, data_files=data_files, model_files=model_files)
