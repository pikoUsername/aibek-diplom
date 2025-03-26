from joblib import load
import pandas as pd


def forecast_sarimax(model_path: str, periods: int, exog_future: pd.DataFrame = None):
	"""
	Загружает модель SARIMAX из model_path и делает прогноз на заданное число периодов.
	Если exog_future предоставлен (DataFrame с будущими значениями экзогенных факторов),
	он используется при прогнозировании.
	Возвращает pandas DataFrame с прогнозом.
	"""
	model = load(model_path)
	forecast_values, conf_int = model.predict(n_periods=periods, exogenous=exog_future, return_conf_int=True)

	# Преобразуем в DataFrame
	forecast_df = pd.DataFrame({
		'forecast': forecast_values,
		'lower_conf_int': conf_int[:, 0],
		'upper_conf_int': conf_int[:, 1]
	})
	return forecast_df