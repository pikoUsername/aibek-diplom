import pandas as pd


def load_and_preprocess(csv_path: str, product: str, freq='MS'):
	"""
	Читает CSV, фильтрует по выбранному товару и приводит данные к временному ряду.
	В CSV должны быть столбцы: date, product, sales, price, ad_budget.
	Возвращает DataFrame с индексом дат и колонками: sales, price, ad_budget.
	"""
	df = pd.read_csv(csv_path)
	df['date'] = pd.to_datetime(df['date'])

	# Фильтрация по выбранному товару
	df = df[df['product'] == product]
	df.set_index('date', inplace=True)
	df = df.asfreq(freq)

	# Заполняем пропуски для числовых данных
	df['sales'] = df['sales'].fillna(method='ffill')
	df['price'] = df['price'].fillna(method='ffill')
	df['ad_budget'] = df['ad_budget'].fillna(method='ffill')

	return df