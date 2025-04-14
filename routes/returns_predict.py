import os
import pandas as pd
import uuid
import pickle
from flask import Blueprint, render_template, request, current_app
import matplotlib.pyplot as plt
from flask_login import login_required, current_user

from db.models import Plot, db

bp = Blueprint("returns_predict", __name__, url_prefix="/predict_returns")

@bp.route("/", methods=["GET", "POST"])
@login_required
def returns_predict():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m[:-4] for m in os.listdir(current_app.config["MODEL_FOLDER"]) if m.endswith(".pkl") and m.startswith("returns_")]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        returns_col = request.form.get("returns_col")
        model_name = request.form.get("model_name")
        forecast_steps = int(request.form.get("forecast_steps", 14))

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        if not os.path.exists(filepath) or not os.path.exists(model_path):
            return render_template("returns_predict.html", message="CSV немесе модель табылмады.", plot_filename=None, data_files=data_files, model_files=model_files)
        df = pd.read_csv(filepath)
        for col in [date_col, returns_col]:
            if col not in df.columns:
                return render_template("returns_predict.html", message=f"Баған {col} жоқ.", plot_filename=None, data_files=data_files, model_files=model_files)
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[returns_col]

        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        model = model_package["model"]

        try:
            forecast = model.get_forecast(steps=forecast_steps).predicted_mean
        except Exception as e:
            return render_template("returns_predict.html", message=f"Болжау қатесі: {e}", plot_filename=None, data_files=data_files, model_files=model_files)

        future_dates = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='D')[1:]
        forecast_series = pd.Series(forecast, index=future_dates)

        plt.figure(figsize=(10, 5))
        plt.plot(ts, label="Тарихи қайтарымдар")
        plt.plot(forecast_series, label="Қайтару болжамы")
        plt.legend()
        plt.title("Қайтару болжамы")
        plot_filename = f"returns_forecast_{csv_filename.split('.')[0]}_{uuid.uuid4().hex[:6]}.png"
        plot_path = os.path.join(current_app.config["PLOT_FOLDER"], plot_filename)
        plt.savefig(plot_path)
        plt.close()

        plot = Plot(user_id=current_user.id, plot_filename=plot_path)

        db.session.add(plot)
        db.session.commit()

        return render_template("returns_predict.html", message="Болжам қалыптасты!", plot_filename=plot_filename, data_files=data_files, model_files=model_files)
    return render_template("returns_predict.html", message=None, plot_filename=None, data_files=data_files, model_files=model_files)
