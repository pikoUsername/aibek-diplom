import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, current_app

bp = Blueprint("trends", __name__, url_prefix="/trends")

@bp.route("/", methods=["GET", "POST"])
def trends():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    message = None
    plot_filename = None

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        category_col = request.form.get("category_col")
        sales_col = request.form.get("sales_col")
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            message = f"Файл {csv_filename} не найден!"
        else:
            df = pd.read_csv(filepath)
            if date_col not in df.columns or category_col not in df.columns or sales_col not in df.columns:
                message = "Один из необходимых столбцов отсутствует!"
            else:
                df[date_col] = pd.to_datetime(df[date_col])
                df.sort_values(by=date_col, inplace=True)
                grouped = df.groupby([category_col, date_col])[sales_col].sum().reset_index()
                plt.figure(figsize=(12,6))
                categories = grouped[category_col].unique()
                for cat in categories:
                    sub = grouped[grouped[category_col] == cat]
                    plt.plot(sub[date_col], sub[sales_col], label=str(cat))
                plt.legend()
                plt.title("Тренды по категориям")
                plot_filename = f"trends_{csv_filename.split('.')[0]}_{np.random.randint(1000)}.png"
                plot_path = os.path.join(current_app.config["PLOT_FOLDER"], plot_filename)
                plt.savefig(plot_path)
                plt.close()
                message = "Тренды построены"
    return render_template("trends.html", message=message, plot_filename=plot_filename, data_files=data_files)
