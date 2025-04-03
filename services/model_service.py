import pickle
from pmdarima.arima import auto_arima

def train_sarima_model(ts, seasonal_period=7):
    """
    Обучает SARIMA-модель на временном ряду ts.
    """
    model = auto_arima(
        ts,
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        d=1,
        start_P=1, start_Q=1,
        max_P=2, max_Q=2,
        D=1,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        approximation=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    model.fit(ts)
    return model


def save_model(model, model_path):
    """
    Сохраняет обученную модель в файл.
    """
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    """
    Загружает модель из файла.
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)
