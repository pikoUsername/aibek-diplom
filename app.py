import os
import pickle
import uuid

import matplotlib
import pandas as pd

matplotlib.use('Agg')  # Чтобы не требовался дисплей (если запускаетесь на сервере)
import matplotlib.pyplot as plt

from flask import Flask, request, render_template
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./data"
app.config["MODEL_FOLDER"] = "./models"
app.config["PLOT_FOLDER"] = "./static/plots"

# Убедимся, что нужные папки существуют
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
os.makedirs(app.config["PLOT_FOLDER"], exist_ok=True)


##############################################################################
# Главная страница (загрузка файла + обзор)
##############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		file = request.files.get("file")
		if file:
			filename = file.filename
			filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
			file.save(filepath)

			# Простейший пример ETL:
			df = pd.read_csv(filepath)
			df.dropna(inplace=True)
			df.to_csv(filepath, index=False)

			return render_template("index.html",
			                       message=f"Файл '{filename}' успешно загружен и сохранён!")
	return render_template("index.html", message=None)


##############################################################################
# Обучение новой SARIMA модели
##############################################################################
@app.route("/train", methods=["GET", "POST"])
def train_sarima():
	if request.method == "POST":
		csv_filename = request.form.get("csv_filename")
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name")

		filepath = os.path.join(app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template("train.html",
			                       message=f"Файл {csv_filename} не найден!")

		# Читаем DataFrame
		df = pd.read_csv(filepath)
		if date_col not in df.columns or target_col not in df.columns:
			return render_template("train.html",
			                       message="Указанных столбцов нет в датафрейме!")

		# Преобразуем столбец с датами
		df[date_col] = pd.to_datetime(df[date_col])
		df.sort_values(by=date_col, inplace=True)
		ts = df.set_index(date_col)[target_col]

		# Обучаем auto_arima
		arima_model = auto_arima(
			ts,
			seasonal=True,
			m=7,  # пример: недельная сезонность
			trace=True,
			error_action='ignore',
			suppress_warnings=True
		)
		arima_model.fit(ts)

		# Сериализуем модель
		model_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		with open(model_path, "wb") as f:
			pickle.dump(arima_model, f)

		return render_template("train.html",
		                       message=f"Модель '{model_name}' успешно обучена и сохранена!")

	return render_template("train.html", message=None)


##############################################################################
# Предсказание и построение графика
##############################################################################
@app.route("/predict", methods=["GET", "POST"])
def predict():
	if request.method == "POST":
		csv_filename = request.form.get("csv_filename")
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name")
		forecast_steps = int(request.form.get("forecast_steps", 14))

		filepath = os.path.join(app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template("predict.html",
			                       message=f"Файл {csv_filename} не найден!",
			                       plot_filename=None)

		model_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		if not os.path.exists(model_path):
			return render_template("predict.html",
			                       message=f"Модель {model_name} не найдена!",
			                       plot_filename=None)

		# Загружаем CSV
		df = pd.read_csv(filepath)
		df[date_col] = pd.to_datetime(df[date_col])
		df.sort_values(by=date_col, inplace=True)
		ts = df.set_index(date_col)[target_col]

		# Загружаем модель
		with open(model_path, "rb") as f:
			arima_model = pickle.load(f)

		# Предсказываем
		forecast = arima_model.predict(n_periods=forecast_steps)

		# Генерируем даты для прогноза
		last_date = ts.index[-1]
		future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='D')[1:]

		# Строим график
		plt.figure(figsize=(10, 5))
		plt.plot(ts, label="Исходные данные")
		plt.plot(future_dates, forecast, color="red", label="Прогноз")
		plt.legend()
		plt.title(f"Прогноз {model_name} на {forecast_steps} шагов")

		# Сохраняем график
		plot_filename = f"plot_{model_name}_{uuid.uuid4().hex}.png"
		plot_path = os.path.join(app.config["PLOT_FOLDER"], plot_filename)
		plt.savefig(plot_path)
		plt.close()

		return render_template("predict.html",
		                       message="Прогноз успешно сформирован!",
		                       plot_filename=plot_filename)

	return render_template("predict.html", message=None, plot_filename=None)


##############################################################################
# Страница метрик (MSE, RMSE, MAPE) с train/test split
##############################################################################
@app.route("/metrics", methods=["GET", "POST"])
def metrics():
	"""
	Берём CSV и выбранную модель,
	разбиваем данные на train/test (например, 80/20 по времени),
	строим прогноз на test, считаем метрики (MSE, RMSE, MAPE).
	"""
	if request.method == "POST":
		csv_filename = request.form.get("csv_filename")
		date_col = request.form.get("date_col")
		target_col = request.form.get("target_col")
		model_name = request.form.get("model_name")
		test_size = float(request.form.get("test_size", 0.2))

		filepath = os.path.join(app.config["UPLOAD_FOLDER"], csv_filename)
		if not os.path.exists(filepath):
			return render_template(
				"metrics.html",
				message=f"Файл {csv_filename} не найден!",
				metrics_dict=None
			)

		model_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
		if not os.path.exists(model_path):
			return render_template(
				"metrics.html",
				message=f"Модель {model_name} не найдена!",
				metrics_dict=None
			)

		# Загружаем данные
		df = pd.read_csv(filepath)
		df[date_col] = pd.to_datetime(df[date_col])
		df.sort_values(by=date_col, inplace=True)
		ts = df.set_index(date_col)[target_col]

		# Делим на train/test по времени
		n = len(ts)
		split_idx = int(n * (1 - test_size))
		train_data = ts.iloc[:split_idx]
		test_data = ts.iloc[split_idx:]

		# Загружаем модель
		with open(model_path, "rb") as f:
			arima_model = pickle.load(f)

		# Переобучаем модель на train, чтобы корректно считать метрики
		arima_model.fit(train_data)

		# Прогнозируем на длину test_data
		forecast_values = arima_model.predict(n_periods=len(test_data))

		# Преобразуем forecast_values в Series с тем же индексом, что и у test_data
		forecast_series = pd.Series(forecast_values, index=test_data.index)

		# Считаем метрики
		mse_value = mean_squared_error(test_data, forecast_series)
		rmse_value = np.sqrt(mse_value)

		# MAPE = mean absolute percentage error
		# (если в test_data нет нулевых значений)
		mape_value = np.mean(np.abs((test_data - forecast_series) / test_data)) * 100

		metrics_dict = {
			"MSE": round(mse_value, 3),
			"RMSE": round(rmse_value, 3),
			"MAPE (%)": round(mape_value, 3)
		}

		return render_template(
			"metrics.html",
			message="Метрики успешно рассчитаны!",
			metrics_dict=metrics_dict
		)

	# Если GET-запрос — просто отрисовываем форму
	return render_template("metrics.html", message=None, metrics_dict=None)


##############################################################################
# История графиков (простейшая реализация — просто показывает PNG в папке)
##############################################################################
@app.route("/history")
def history():
	plots = os.listdir(app.config["PLOT_FOLDER"])
	return render_template("history.html", plots=plots)


if __name__ == "__main__":
	app.run(debug=True)
