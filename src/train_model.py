import pmdarima as pm
from joblib import dump
from datetime import datetime
from src.data_prep import load_and_preprocess
import matplotlib.pyplot as plt
import os


def train_sarimax(csv_path: str, product: str, model_dir: str, freq='MS', test_size=3):
	"""
	Обучает модель SARIMAX на данных выбранного товара, используя экзогенные факторы.
	Сохраняет модель с версией (датой) и возвращает путь к сохранённой модели.
	"""
	df = load_and_preprocess(csv_path, product, freq=freq)
	# Делим данные на train и test (по времени)
	train_data = df.iloc[:-test_size]
	# Опорная переменная
	y_train = train_data['sales']
	# Экзогенные перементы: цена и рекламный бюджет
	exog_train = train_data[['price', 'ad_budget']]

	# Используем auto_arima с учетом экзогенных факторов
	model = pm.auto_arima(
		y_train,
		exogenous=exog_train,
		seasonal=True,
		m=12,  # годовая сезонность для месячных данных
		trace=True,
		error_action='ignore',
		suppress_warnings=True,
		stepwise=True
	)

	# Сохраняем модель с версией (дата)
	version = datetime.now().strftime("%Y%m%d_%H%M%S")
	model_filename = f"sarimax_model_{product}_{version}.pkl"
	model_path = os.path.join(model_dir, model_filename)
	dump(model, model_path)

	# (Опционально) можно сохранить диагностические графики остатков
	diagnostics_path = os.path.join(model_dir, f"diagnostics_{product}_{version}.png")
	plt.figure()
	model.plot_diagnostics(figsize=(10, 8))
	plt.tight_layout()
	plt.savefig(diagnostics_path)
	plt.close()

	return model_path, diagnostics_path