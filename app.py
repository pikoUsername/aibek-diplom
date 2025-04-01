import os
import pickle
import uuid

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, redirect, url_for
import numpy as np

###############################################################################
# ИНИЦИАЛИЗАЦИЯ FLASK
###############################################################################
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "./data"
app.config["MODEL_FOLDER"] = "./models"
app.config["PLOT_FOLDER"] = "./static/plots"
app.config["SCRIPT_FOLDER"] = "./scripts"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
os.makedirs(app.config["PLOT_FOLDER"], exist_ok=True)
os.makedirs(app.config["SCRIPT_FOLDER"], exist_ok=True)


###############################################################################
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ПРИМЕНЯЕМ ПОЛЬЗОВАТЕЛЬСКИЙ СКРИПТ
###############################################################################
def apply_transform(df: pd.DataFrame, script_path: str) -> pd.DataFrame:
	if not os.path.exists(script_path):
		return df

	# хотим дать скрипту доступ к pd, np, и т.д.
	import numpy as np
	global_env = {
		"__builtins__": __builtins__,  # чтобы скрипт мог сам import-ить всё подряд
		"pd": pd,
		"np": np,
		# если хотите, можно добавлять ещё что-то:
		# "math": math,
		# "datetime": datetime,
		# ...
	}
	local_env = {"df": df}  # локальное окружение — куда запишется обновлённая df

	with open(script_path, "r", encoding="utf-8") as f:
		code = f.read()

	exec(code, global_env, local_env)

	return local_env.get("df", df)


def get_script_list():
	"""
	Получаем список всех .py-файлов в папке scripts (чтобы пользователь мог выбрать).
	"""
	return [f for f in os.listdir(app.config["SCRIPT_FOLDER"]) if f.endswith(".py")]


###############################################################################
# ГЛАВНАЯ СТРАНИЦА: ЗАГРУЗКА CSV
###############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		file = request.files.get("file")
		if file:
			filename = file.filename
			filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
			file.save(filepath)
			# Простая ETL
			df = pd.read_csv(filepath)
			df.dropna(inplace=True)
			df.to_csv(filepath, index=False)

			return render_template("index.html",
			                       message=f"Файл '{filename}' успешно загружен!")
	return render_template("index.html", message=None)


###############################################################################
# РОУТ ДЛЯ ЗАГРУЗКИ СКРИПТА
###############################################################################
@app.route("/upload_script", methods=["GET", "POST"])
def upload_script():
	"""
	Пользователь может вставить Python-код, который будет сохраняться как .py-файл.
	"""
	if request.method == "POST":
		script_code = request.form.get("script_code")
		script_name = request.form.get("script_name")

		# Если не указано имя, генерируем
		if not script_name:
			script_name = f"script_{uuid.uuid4().hex}"
		if not script_name.endswith(".py"):
			script_name += ".py"

		script_path = os.path.join(app.config["SCRIPT_FOLDER"], script_name)
		with open(script_path, "w", encoding="utf-8") as f:
			f.write(script_code)

		return render_template("upload_script.html",
		                       message=f"Скрипт '{script_name}' сохранён!")

	return render_template("upload_script.html", message=None)


###############################################################################
# ОБУЧЕНИЕ SARIMA (/train)
###############################################################################
@app.route("/train", methods=["GET", "POST"])
def train_sarima():
	data_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
	script_files = get_script_list()  # список .py-скриптов

	if request.method == "POST":
		csv_filename = request.form.get("csv_filename")
		script_name = request.form.get("script_name")  # выбранный скрипт
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name", f"model_{uuid.uuid4().hex}")

		filepath = os.path.join(app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template("train.html",
			                       message=f"Файл {csv_filename} не найден!",
			                       data_files=data_files,
			                       script_files=script_files)

		# Загружаем CSV
		df = pd.read_csv(filepath)
		# Применяем пользовательский скрипт, если выбран
		if script_name and script_name != "none":
			script_path = os.path.join(app.config["SCRIPT_FOLDER"], script_name)
			df = apply_transform(df, script_path)

		# Проверяем нужные столбцы
		if date_col not in df.columns or target_col not in df.columns:
			return render_template("train.html",
			                       message=f"Нет {date_col} или {target_col} в данных!",
			                       data_files=data_files,
			                       script_files=script_files)

		# Приводим дату к datetime, упорядочиваем
		df[date_col] = pd.to_datetime(df[date_col])
		df.sort_values(by=date_col, inplace=True)
		ts = df.set_index(date_col)[target_col]

		# Пробуем "заставить" auto_arima искать более сложную модель
		from pmdarima.arima import auto_arima

		arima_model = auto_arima(
			ts,
			start_p=1,  # начальные значения p, q
			start_q=1,
			max_p=3,  # максимально 3
			max_q=3,
			d=1,  # форсируем хотя бы первую разность
			start_P=1,  # для сезонной части
			start_Q=1,
			max_P=2,
			max_Q=2,
			D=1,  # форсируем сезонную разность
			seasonal=True,
			m=7,  # если данные ежедневные и есть недельная сезонность
			stepwise=True,  # пусть пошагово перебирает
			approximation=False,
			trace=True,  # чтобы видеть в консоли, что он перебирает
			error_action='ignore',
			suppress_warnings=True
		)
		arima_model.fit(ts)

		# Сохраняем модель
		model_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		with open(model_path, "wb") as f:
			pickle.dump(arima_model, f)

		return render_template("train.html",
		                       message=f"Модель '{model_name}' обучена и сохранена!",
		                       data_files=data_files,
		                       script_files=script_files)

	# GET
	return render_template("train.html",
	                       message=None,
	                       data_files=data_files,
	                       script_files=script_files)


###############################################################################
# ПРЕДСКАЗАНИЕ (/predict)
###############################################################################
@app.route("/predict", methods=["GET", "POST"])
def predict():
	data_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
	model_files = [m[:-4] for m in os.listdir(app.config["MODEL_FOLDER"]) if m.endswith(".pkl")]
	script_files = get_script_list()

	if request.method == "POST":
		csv_filename = request.form.get("csv_filename")
		script_name = request.form.get("script_name")
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name")
		forecast_steps = int(request.form.get("forecast_steps", 14))

		filepath = os.path.join(app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template("predict.html",
			                       message=f"Файл {csv_filename} не найден!",
			                       plot_filename=None,
			                       data_files=data_files,
			                       model_files=model_files,
			                       script_files=script_files)

		model_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		if not os.path.exists(model_path):
			return render_template("predict.html",
			                       message=f"Модель {model_name} не найдена!",
			                       plot_filename=None,
			                       data_files=data_files,
			                       model_files=model_files,
			                       script_files=script_files)

		# Загружаем CSV
		df = pd.read_csv(filepath)
		# Применяем пользовательский скрипт
		if script_name and script_name != "none":
			script_path = os.path.join(app.config["SCRIPT_FOLDER"], script_name)
			df = apply_transform(df, script_path)

		# Проверка столбцов
		if date_col not in df.columns or target_col not in df.columns:
			return render_template("predict.html",
			                       message=f"Нет {date_col}/{target_col} в данных!",
			                       plot_filename=None,
			                       data_files=data_files,
			                       model_files=model_files,
			                       script_files=script_files)

		df[date_col] = pd.to_datetime(df[date_col])
		df.sort_values(by=date_col, inplace=True)
		ts = df.set_index(date_col)[target_col]

		# Загружаем модель
		with open(model_path, "rb") as f:
			arima_model = pickle.load(f)

		# Предсказываем
		forecast_values = arima_model.predict(n_periods=forecast_steps)
		last_date = ts.index[-1]
		future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='D')[1:]
		forecast_series = pd.Series(forecast_values, index=future_dates)

		# Рисуем график
		plt.figure(figsize=(10, 5))
		plt.plot(ts, label="Исторические")
		plt.plot(forecast_series, color="red", label="Прогноз")
		plt.legend()
		plt.title(f"Прогноз {model_name}")

		plot_filename = f"plot_{model_name}_{uuid.uuid4().hex}.png"
		plot_path = os.path.join(app.config["PLOT_FOLDER"], plot_filename)
		plt.savefig(plot_path)
		plt.close()

		csv_filename = request.form.get("csv_filename")
		script_name = request.form.get("script_name")
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name")
		forecast_steps = request.form.get("forecast_steps", 14)

		# ... (логика предсказания)

		# В конце — передаём ВСЕ значения обратно, чтобы шаблон мог
		# заполнить форму как раньше:
		return render_template("predict.html",
		                       message="Прогноз успешно сформирован!",
		                       plot_filename=plot_filename,
		                       data_files=data_files,
		                       model_files=model_files,
		                       script_files=script_files,
		                       # <--- добавляем «выбранные» значения:
		                       selected_csv=csv_filename,
		                       selected_script=script_name,
		                       selected_model=model_name,
		                       entered_date_col=date_col,
		                       entered_target_col=target_col,
		                       entered_steps=forecast_steps)

	# GET
	return render_template("predict.html",
	                       message=None,
	                       plot_filename=None,
	                       data_files=data_files,
	                       model_files=model_files,
	                       script_files=script_files,
	                       selected_csv=None,
	                       selected_script=None,
	                       selected_model=None,
	                       entered_date_col=None,
	                       entered_target_col=None,
	                       entered_steps=None)


###############################################################################
# МЕТРИКИ (/metrics)
###############################################################################
@app.route("/metrics", methods=["GET", "POST"])
def metrics():
	# Собираем списки файлов
	data_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
	model_files = [m[:-4] for m in os.listdir(app.config["MODEL_FOLDER"]) if m.endswith(".pkl")]
	script_files = get_script_list()  # функция, которая возвращает список .py из SCRIPT_FOLDER

	if request.method == "POST":
		# 1) Считываем поля формы
		csv_filename = request.form.get("csv_filename")
		script_name = request.form.get("script_name")  # имя скрипта (.py) или "none"
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name")
		test_size = float(request.form.get("test_size", 0.2))

		# 2) Проверяем наличие CSV
		filepath = os.path.join(app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template(
				"metrics.html",
				message=f"Файл {csv_filename} не найден!",
				metrics_dict=None,
				data_files=data_files,
				model_files=model_files,
				script_files=script_files
			)

		# 3) Проверяем наличие модели
		model_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		if not os.path.exists(model_path):
			return render_template(
				"metrics.html",
				message=f"Модель {model_name} не найдена!",
				metrics_dict=None,
				data_files=data_files,
				model_files=model_files,
				script_files=script_files
			)

		# 4) Загружаем CSV
		df = pd.read_csv(filepath)
		print("[DEBUG] CSV loaded. df.shape =", df.shape)

		# 5) Применяем пользовательский скрипт (если выбран)
		if script_name and script_name != "none":
			script_path = os.path.join(app.config["SCRIPT_FOLDER"], script_name)
			df_before = df.copy()
			df = apply_transform(df, script_path)
			print("[DEBUG] Script applied:", script_name)
			print("     Before:", df_before.shape, "After:", df.shape)

		# 6) Проверяем наличие нужных столбцов
		if date_col not in df.columns or target_col not in df.columns:
			return render_template(
				"metrics.html",
				message=f"Нет {date_col} или {target_col} в данных!",
				metrics_dict=None,
				data_files=data_files,
				model_files=model_files,
				script_files=script_files
			)

		# 7) Преобразуем даты, сортируем
		df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
		df.sort_values(by=date_col, inplace=True)
		print(f"[DEBUG] date_col='{date_col}' parsed, sorted by date.")

		# Убедимся, что нет NaN в датах:
		df = df.dropna(subset=[date_col])
		print("[DEBUG] Dropped rows with NaN in date_col. df.shape =", df.shape)

		# 8) Формируем Series
		ts = df.set_index(date_col)[target_col]
		print("[DEBUG] Final ts.shape =", ts.shape)

		# 9) Разделяем на train/test
		n = len(ts)
		split_idx = int(n * (1 - test_size))
		train_data = ts.iloc[:split_idx]
		test_data = ts.iloc[split_idx:]

		print(f"[DEBUG] train_data: {train_data.shape}, test_data: {test_data.shape}")

		# 10) Загружаем модель
		with open(model_path, "rb") as f:
			arima_model = pickle.load(f)
		print("[DEBUG] Model loaded:", model_name)

		# 11) Переобучаем на train
		arima_model.fit(train_data)
		print("[DEBUG] Model re-fitted on train_data")

		# 12) Прогнозируем на длину test_data
		forecast_vals = arima_model.predict(n_periods=len(test_data))
		print("[DEBUG] forecast_vals len:", len(forecast_vals))
		print("[DEBUG] test_data len:", len(test_data))

		# Формируем Series с тем же индексом, что у test_data
		forecast_series = pd.Series(forecast_vals, index=test_data.index)

		# 13) Отладочные принты индексов
		print("[DEBUG] test_data index:", test_data.index)
		print("[DEBUG] forecast_series index:", forecast_series.index)

		common_index = test_data.index.intersection(forecast_series.index)
		print("[DEBUG] Intersection index length:", len(common_index))

		# 14) Можно (необязательно) явно выровнять и удалить NaN
		test_data = test_data.loc[common_index].dropna()
		forecast_series = forecast_series.loc[common_index].dropna()

		print("[DEBUG] After alignment, len(test_data) =", len(test_data))
		print("[DEBUG] After alignment, len(forecast_series) =", len(forecast_series))

		if len(test_data) == 0 or len(forecast_series) == 0:
			return render_template(
				"metrics.html",
				message="Нет валидных точек для расчёта метрик (пересечение пусто или NaN).",
				metrics_dict=None,
				data_files=data_files,
				model_files=model_files,
				script_files=script_files
			)

		# 15) Считаем метрики
		from sklearn.metrics import mean_squared_error
		mse_value = mean_squared_error(test_data, forecast_series)
		rmse_value = np.sqrt(mse_value)

		# (Если есть нули в test_data, MAPE может быть некорректен)
		mape_value = np.mean(np.abs((test_data - forecast_series) / test_data)) * 100

		metrics_dict = {
			"MSE": round(mse_value, 3),
			"RMSE": round(rmse_value, 3),
			"MAPE (%)": round(mape_value, 3)
		}

		return render_template(
			"metrics.html",
			message="Метрики успешно рассчитаны!",
			metrics_dict=metrics_dict,
			data_files=data_files,
			model_files=model_files,
			script_files=script_files
		)

	# GET-запрос -> форма
	return render_template(
		"metrics.html",
		message=None,
		metrics_dict=None,
		data_files=data_files,
		model_files=model_files,
		script_files=script_files
	)


###############################################################################
# ИСТОРИЯ ГРАФИКОВ (/history)
###############################################################################
@app.route("/history")
def history():
	plots = os.listdir(app.config["PLOT_FOLDER"])
	return render_template("history.html", plots=plots)


@app.route("/file_manager", methods=["GET"])
def file_manager():
	# Собираем списки файлов
	data_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if
	              os.path.isfile(os.path.join(app.config["UPLOAD_FOLDER"], f))]
	model_files = [f for f in os.listdir(app.config["MODEL_FOLDER"]) if
	               os.path.isfile(os.path.join(app.config["MODEL_FOLDER"], f))]
	script_files = [f for f in os.listdir(app.config["SCRIPT_FOLDER"]) if
	                os.path.isfile(os.path.join(app.config["SCRIPT_FOLDER"], f))]

	# Опционально — если хотите ещё и картинки из ./static/plots
	plot_files = [f for f in os.listdir(app.config["PLOT_FOLDER"]) if
	              os.path.isfile(os.path.join(app.config["PLOT_FOLDER"], f))]

	return render_template("file_manager.html",
	                       data_files=data_files,
	                       model_files=model_files,
	                       script_files=script_files,
	                       plot_files=plot_files)


# ---------------------------------------------------------------------
# Маршрут: удаляем файл в нужной папке, по типу
# ---------------------------------------------------------------------
@app.route("/delete_file", methods=["GET"])
def delete_file():
	file_type = request.args.get("file_type")  # data / model / script / plot
	filename = request.args.get("filename")
	if not file_type or not filename:
		return "Не указан file_type или filename", 400

	# Определяем папку по типу
	if file_type == "data":
		folder = app.config["UPLOAD_FOLDER"]
	elif file_type == "model":
		folder = app.config["MODEL_FOLDER"]
	elif file_type == "script":
		folder = app.config["SCRIPT_FOLDER"]
	elif file_type == "plot":
		folder = app.config["PLOT_FOLDER"]
	else:
		return "Неверный тип файла", 400

	filepath = os.path.join(folder, filename)
	if os.path.exists(filepath):
		os.remove(filepath)
		return redirect(url_for("file_manager"))
	else:
		return f"Файл {filename} не найден в {folder}", 404


if __name__ == "__main__":
	app.run(debug=True)
