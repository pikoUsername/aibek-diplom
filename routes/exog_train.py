import os
import pandas as pd
import uuid
import pickle
import json
from flask import Blueprint, render_template, request, current_app
from flask_login import login_required
from statsmodels.tsa.statespace.sarimax import SARIMAX

from services.login_manager import login_manager

bp = Blueprint("exog_train", __name__, url_prefix="/train_exog")

@bp.route("/", methods=["GET", "POST"])
@login_required
def exog_train():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        exog_cols_str = request.form.get("exog_cols", "").strip()
        model_name = request.form.get("model_name", f"exog_{uuid.uuid4().hex}")
        order_str = request.form.get("order", "").strip()
        seasonal_order_str = request.form.get("seasonal_order", "").strip()
        additional_params_str = request.form.get("additional_params", "{}").strip()

        try:
            order = tuple(int(x.strip()) for x in order_str.split(",")) if order_str else (1,1,1)
            if len(order) != 3: raise ValueError
        except Exception:
            return render_template("exog_train.html", message="Қате order форматы.", data_files=data_files)
        try:
            seasonal_order = tuple(int(x.strip()) for x in seasonal_order_str.split(",")) if seasonal_order_str else (1,1,1,7)
            if len(seasonal_order) != 4: raise ValueError
        except Exception:
            return render_template("exog_train.html", message="Қате seasonal_order форматы.", data_files=data_files)
        try:
            additional_params = json.loads(additional_params_str)
        except Exception as e:
            return render_template("exog_train.html", message=f"Қосымша опциялардың дұрыс емес форматы: {e}", data_files=data_files)

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        if not os.path.exists(filepath):
            return render_template("exog_train.html", message=f"Файл {csv_filename} табылмады!", data_files=data_files)
        df = pd.read_csv(filepath)
        for col in [date_col, target_col]:
            if col not in df.columns:
                return render_template("exog_train.html", message=f"Жол {col} табылмады.", data_files=data_files)
        if exog_cols_str:
            exog_cols = [col.strip() for col in exog_cols_str.split(",") if col.strip()]
            missing = [col for col in exog_cols if col not in df.columns]
            if missing:
                return render_template("exog_train.html", message=f"Экзогендік бағандар жоқ: {missing}", data_files=data_files)
            exog_data = df[exog_cols]
        else:
            exog_cols = None
            exog_data = None

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[target_col]

        try:
            model = SARIMAX(ts, exog=exog_data, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                            **additional_params).fit(disp=False)
        except Exception as e:
            return render_template("exog_train.html", message=f"Модельді оқыту қатесі: {e}", data_files=data_files)

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

        return render_template("exog_train.html", message=f"Модель '{model_name}' оқытылды және сақталды!", data_files=data_files)
    return render_template("exog_train.html", message=None, data_files=data_files)
