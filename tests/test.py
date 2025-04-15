import os
import tempfile
import uuid
import pytest
import pandas as pd
import numpy as np
import matplotlib

from db.user_repo import create_user, get_user_by_email, get_all_users
from helpers.plot_utils import save_forecast_plot
from helpers.script_utils import apply_transform, get_script_list, apply_transform_code
from services.metrics_service import calculate_forecast_metrics
from services.model_service import train_sarima_model, save_model, load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

# =====================================================
# Фикстура для временной директории
# =====================================================

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

# =====================================================
# Тестирование функции построения графика прогнозов
# =====================================================

def test_save_forecast_plot(temp_dir):
    # Создаем dummy time series
    ts = pd.Series(np.arange(10))
    forecast_series = pd.Series(np.arange(10, 15), index=np.arange(10, 15))
    model_name = "test_model"
    plot_filename = save_forecast_plot(ts, forecast_series, model_name, temp_dir)
    filepath = os.path.join(temp_dir, plot_filename)
    assert os.path.exists(filepath)
    assert plot_filename.endswith(".png")

# =====================================================
# Тестирование apply_transform (применение скрипта из файла)
# =====================================================

def test_apply_transform(temp_dir):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    # Скрипт, который добавляет колонку C = A + B
    script_content = "df['C'] = df['A'] + df['B']"
    script_path = os.path.join(temp_dir, "transform.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    df_transformed = apply_transform(df.copy(), script_path)
    assert "C" in df_transformed.columns
    pd.testing.assert_series_equal(df_transformed["C"], df["A"] + df["B"])

# =====================================================
# Тестирование get_script_list (получение списка .py файлов)
# =====================================================

def test_get_script_list(temp_dir):
    filenames = ["a.py", "b.py", "c.txt", "d.py"]
    for name in filenames:
        with open(os.path.join(temp_dir, name), "w") as f:
            f.write("print('Hello')")
    script_list = get_script_list(temp_dir)
    expected = {"a.py", "b.py", "d.py"}
    assert set(script_list) == expected

# =====================================================
# Тестирование apply_transform_code (применение кода к DataFrame)
# =====================================================

def test_apply_transform_code():
    df = pd.DataFrame({"X": [10, 20, 30]})
    code = "df['Y'] = df['X'] / 2"
    df_transformed = apply_transform_code(df.copy(), code)
    assert "Y" in df_transformed.columns
    np.testing.assert_allclose(df_transformed["Y"].values, np.array([5, 10, 15]))

# =====================================================
# Тестирование calculate_forecast_metrics
# =====================================================

def test_calculate_forecast_metrics():
    actual = np.array([10, 20, 30, 40])
    predicted = np.array([12, 18, 33, 37])
    metrics = calculate_forecast_metrics(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    assert np.isclose(metrics["MSE"], mse, atol=1e-3)
    assert np.isclose(metrics["RMSE"], rmse, atol=1e-3)
    assert np.isclose(metrics["MAPE (%)"], mape, atol=1e-3)

# =====================================================
# Тестирование train_sarima_model
# =====================================================

def test_train_sarima_model():
    # Синтетический временной ряд (небольшой)
    ts = pd.Series(np.sin(np.linspace(0, 2*np.pi, 50)) + np.random.normal(0, 0.1, 50))
    model = train_sarima_model(ts, seasonal_period=7)
    # Проверяем, что модель объект и у нее есть атрибут aic
    assert model is not None
    assert hasattr(model, "aic")

# =====================================================
# Тестирование сохранения/загрузки модели
# =====================================================

def test_save_and_load_model(temp_dir):
    model_dummy = {"param": 42}
    model_path = os.path.join(temp_dir, "dummy_model.pkl")
    save_model(model_dummy, model_path)
    loaded = load_model(model_path)
    assert loaded == model_dummy

# =====================================================
# Тестирование функций работы с пользователями (с использованием dummy session)
# =====================================================

# Создадим dummy модели и сессию для имитации SQLAlchemy
class DummyUser:
    def __init__(self, id, username, email, password_hash):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash

class DummyQuery:
    def __init__(self, items):
        self.items = items
    def filter_by(self, **kwargs):
        filtered = [item for item in self.items if all(getattr(item, k) == v for k, v in kwargs.items())]
        return DummyQuery(filtered)
    def first(self):
        return self.items[0] if self.items else None
    def all(self):
        return self.items

class DummySession:
    def __init__(self):
        self.users = []
    def add(self, obj):
        self.users.append(obj)
    def commit(self):
        pass
    def query(self, model):
        # Будем игнорировать model, возвращая DummyQuery по нашим пользователям
        return DummyQuery(self.users)

@pytest.fixture
def dummy_session():
    return DummySession()

def test_user_functions(dummy_session):
    # Тестируем create_user, get_user_by_email, get_all_users
    user = create_user(dummy_session, "testuser", "test@example.com", "hashed_pwd")
    fetched = get_user_by_email(dummy_session, "test@example.com")
    assert fetched is not None
    assert fetched.username == "testuser"
    all_users = get_all_users(dummy_session)
    assert len(all_users) == 1
