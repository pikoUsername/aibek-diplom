import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_forecast_metrics(actual, predicted):
    """
    Вычисляет MSE, RMSE и MAPE между фактическими и предсказанными значениями.
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 3)
    }
