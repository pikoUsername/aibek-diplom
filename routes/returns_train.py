import os
import pandas as pd
import uuid
import pickle
import json
from flask import Blueprint, render_template, request, current_app
from statsmodels.tsa.statespace.sarimax import SARIMAX
from helpers.script_utils import apply_transform, get_script_list

bp = Blueprint("returns_train", __name__, url_prefix="/train_returns")


@bp.route("/", methods=["GET", "POST"])
def returns_train():
	data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
	script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])

	if request.method == "POST":
		csv_filename = request.form.get("csv_filename")
		script_name = request.form.get("script_name")
		date_col = request.form.get("date_col")
		returns_col = request.form.get("returns_col")
		model_name = request.form.get("model_name", f"returns_model_{uuid.uuid4().hex}")

		order_str = request.form.get("order", "").strip()
		seasonal_order_str = request.form.get("seasonal_order", "").strip()
		additional_params_str = request.form.get("additional_params", "{}").strip()

		if order_str:
			try:
				order = tuple(int(x.strip()) for x in order_str.split(","))
				if len(order) != 3:
					raise ValueError
			except Exception:
				return render_template("returns_train.html", message="Неверный формат order (ожидается p,d,q)",
				                       data_files=data_files, script_files=script_files)
		else:
			order = (1, 1, 1)

		if seasonal_order_str:
			try:
				seasonal_order = tuple(int(x.strip()) for x in seasonal_order_str.split(","))
				if len(seasonal_order) != 4:
					raise ValueError
			except Exception:
				return render_template("returns_train.html",
				                       message="Неверный формат seasonal_order (ожидается P,D,Q,m)",
				                       data_files=data_files, script_files=script_files)
		else:
			seasonal_order = (1, 1, 1, 7)

		try:
			additional_params = json.loads(additional_params_str)
		except Exception as e:
			return render_template("returns_train.html", message=f"Неверный формат дополнительных параметров: {e}",
			                       data_files=data_files, script_files=script_files)

		filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template("returns_train.html", message=f"Файл {csv_filename} не найден!",
			                       data_files=data_files, script_files=script_files)

		df = pd.read_csv(filepath)
		if script_name and script_name != "none":
			script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
			df = apply_transform(df, script_path)

		for col in [date_col, returns_col]:
			if col not in df.columns:
				return render_template("returns_train.html", message=f"Столбец {col} отсутствует!",
				                       data_files=data_files, script_files=script_files)

		df[date_col] = pd.to_datetime(df[date_col])
		df.sort_values(by=date_col, inplace=True)
		ts = df.set_index(date_col)[returns_col]

		try:
			model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
			                enforce_stationarity=False, enforce_invertibility=False,
			                **additional_params).fit(disp=False)
		except Exception as e:
			return render_template("returns_train.html", message=f"Ошибка обучения модели: {e}",
			                       data_files=data_files, script_files=script_files)

		model_package = {
			"model": model,
			"order": order,
			"seasonal_order": seasonal_order,
			"additional_params": additional_params
		}
		model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		with open(model_path, "wb") as f:
			pickle.dump(model_package, f)

		return render_template("returns_train.html", message=f"Модель '{model_name}' обучена и сохранена!",
		                       data_files=data_files, script_files=script_files)

	return render_template("returns_train.html", message=None,
	                       data_files=data_files, script_files=script_files)
