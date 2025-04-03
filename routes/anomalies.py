import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, current_app

bp = Blueprint("anomalies", __name__, url_prefix="/anomalies")

def detect_anomalies_zscore(series, threshold=3):
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    return series[np.abs(z_scores) > threshold]

@bp.route("/", methods=["GET", "POST"])
def anomalies():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    message = None
    plot_filename = None

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            message = f"Файл {csv_filename} не найден!"
        else:
            df = pd.read_csv(filepath)
            if date_col not in df.columns or target_col not in df.columns:
                message = f"Столбец {date_col} или {target_col} отсутствует в данных!"
            else:
                df[date_col] = pd.to_datetime(df[date_col])
                df.sort_values(by=date_col, inplace=True)
                ts = df.set_index(date_col)[target_col]
                anomalies_series = detect_anomalies_zscore(ts)
                plt.figure(figsize=(10,5))
                plt.plot(ts, label="Данные")
                if not anomalies_series.empty:
                    plt.scatter(anomalies_series.index, anomalies_series, color="red", label="Аномалии")
                plt.legend()
                plt.title("Аномалии / Всплески")
                plot_filename = f"anomalies_{csv_filename.split('.')[0]}_{np.random.randint(1000)}.png"
                plot_path = os.path.join(current_app.config["PLOT_FOLDER"], plot_filename)
                plt.savefig(plot_path)
                plt.close()
                message = "Аномалии обнаружены"
    return render_template("anomalies.html", message=message, plot_filename=plot_filename, data_files=data_files)
