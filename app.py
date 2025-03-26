# app.py
import json
import os

import pandas as pd
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, request, redirect, url_for, flash

from src.data_prep import load_and_preprocess
from src.economic import calculate_optimal_order
from src.predict import forecast_sarimax
from src.train_model import train_sarimax

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'data/raw'
MODEL_FOLDER = 'models'

# Убедимся, что папки существуют
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Глобально сохраняем путь к загруженному файлу и выбранный товар
app.config['LAST_UPLOADED_CSV'] = None
app.config['SELECTED_PRODUCT'] = None
app.config['LAST_MODEL_PATH'] = None
app.config['LAST_DIAGNOSTICS_PATH'] = None


@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		# Обработка загрузки файла
		if 'csv_file' not in request.files:
			flash('Файл не найден в запросе.')
			return redirect(request.url)
		file = request.files['csv_file']
		if file.filename == '':
			flash('Файл не выбран.')
			return redirect(request.url)
		filepath = os.path.join(UPLOAD_FOLDER, file.filename)
		file.save(filepath)
		app.config['LAST_UPLOADED_CSV'] = filepath
		flash(f'Файл {file.filename} успешно загружен!')

		# Если в файле есть колонка product, можно извлечь уникальные товары
		df = pd.read_csv(filepath)
		products = df['product'].unique().tolist()
		return render_template('index.html', products=products)
	return render_template('index.html', products=[])


# Обучение модели
@app.route('/train', methods=['POST'])
def train():
	product = request.form.get('product')
	if not product:
		flash('Выберите товар для обучения модели.')
		return redirect(url_for('index'))
	app.config['SELECTED_PRODUCT'] = product

	csv_path = app.config.get('LAST_UPLOADED_CSV')
	if not csv_path:
		flash('Сначала загрузите файл с данными.')
		return redirect(url_for('index'))

	# Обучаем модель с экзогенными факторами
	model_path, diagnostics_path = train_sarimax(csv_path, product, MODEL_FOLDER, freq='MS', test_size=3)
	app.config['LAST_MODEL_PATH'] = model_path
	app.config['LAST_DIAGNOSTICS_PATH'] = diagnostics_path
	flash(f'Модель для товара "{product}" обучена и сохранена: {model_path}')
	flash(f'Диагностический график сохранён: {diagnostics_path}')
	return redirect(url_for('index'))


# Прогнозирование
@app.route('/forecast', methods=['POST'])
def forecast():
	periods_str = request.form.get('periods', '6')
	try:
		periods = int(periods_str)
	except:
		flash('Некорректное число периодов!')
		return redirect(url_for('index'))

	model_path = app.config.get('LAST_MODEL_PATH')
	if not model_path:
		flash('Сначала обучите модель!')
		return redirect(url_for('index'))

	# Если у пользователя есть прогнозные данные для экзогенных факторов, их можно обработать.
	# Здесь для простоты заполним средними значениями из обучающих данных.
	csv_path = app.config.get('LAST_UPLOADED_CSV')
	product = app.config.get('SELECTED_PRODUCT')
	df = load_and_preprocess(csv_path, product, freq='MS')
	# Берём последние известные значения
	last_exog = df[['price', 'ad_budget']].iloc[-1:]
	exog_future = pd.concat([last_exog] * periods, ignore_index=True)

	forecast_df = forecast_sarimax(model_path, periods, exog_future)

	# Получаем экономическую интерпретацию для первого прогноза (как пример)
	first_forecast = forecast_df['forecast'].iloc[0]
	recommended_order = calculate_optimal_order(first_forecast, safety_factor=0.1)

	# Готовим график (история + прогноз)
	history = df['sales']
	last_date = df.index[-1]
	future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
	forecast_df.index = future_dates

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=history.index.astype(str), y=history.values,
	                         mode='lines+markers', name='История'))
	fig.add_trace(go.Scatter(x=forecast_df.index.astype(str), y=forecast_df['forecast'], mode='lines+markers', name='Прогноз', line=dict(dash='dash')))
	fig.update_layout(
		title=f'Прогноз спроса для {product}',
		xaxis_title='Дата',
		yaxis_title='Продажи')
	graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template(
		'forecast.html',
		forecast_table=forecast_df.to_html(classes='table table-striped'),
		graph_json=graph_json,
		recommended_order=recommended_order)


# Страница диагностики модели
@app.route('/diagnostics')
def diagnostics():
	diagnostics_path = app.config.get('LAST_DIAGNOSTICS_PATH')
	if diagnostics_path and os.path.exists(diagnostics_path):
		# Покажем диагностический график как картинку
		# Для простоты отправим ссылку на этот файл, если он доступен через static или отдельный маршрут
		return render_template('diagnostics.html', diag_image=diagnostics_path)
	else:
		flash('Диагностический график не найден.')
		return redirect(url_for('index'))


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)
