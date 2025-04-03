import os
import uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_forecast_plot(ts, forecast_series, model_name, plot_folder):
    """
    Строит и сохраняет график прогноза.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(ts, label="Исторические данные")
    plt.plot(forecast_series, color="red", label="Прогноз")
    plt.title(f"Прогноз: {model_name}")
    plt.legend()

    plot_filename = f"plot_{model_name}_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(plot_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_filename
