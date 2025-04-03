import os
import pandas as pd
import uuid
import pickle
from flask import Blueprint, render_template, request, current_app
from helpers.script_utils import apply_transform, get_script_list
from helpers.plot_utils import save_forecast_plot

bp = Blueprint("predict", __name__, url_prefix="/predict")

@bp.route("/", methods=["GET", "POST"])
def predict():
    data_files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")]
    model_files = [m[:-4] for m in os.listdir(current_app.config["MODEL_FOLDER"]) if m.endswith(".pkl")]
    script_files = get_script_list(current_app.config["SCRIPT_FOLDER"])

    if request.method == "POST":
        csv_filename = request.form.get("csv_filename")
        script_name = request.form.get("script_name")
        date_col = request.form.get("date_col")
        target_col = request.form.get("target_col")
        model_name = request.form.get("model_name")
        forecast_steps = int(request.form.get("forecast_steps", 14))

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_filename)
        model_path = os.path.join(current_app.config["MODEL_FOLDER"], f"{model_name}.pkl")

        if not os.path.exists(filepath):
            return render_template("predict.html", message=f"Файл {csv_filename} не найден!",
                                   plot_filename=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        if not os.path.exists(model_path):
            return render_template("predict.html", message=f"Модель {model_name} не найдена!",
                                   plot_filename=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        df = pd.read_csv(filepath)
        if script_name and script_name != "none":
            script_path = os.path.join(current_app.config["SCRIPT_FOLDER"], script_name)
            df = apply_transform(df, script_path)

        if date_col not in df.columns or target_col not in df.columns:
            return render_template("predict.html", message=f"Нет {date_col}/{target_col} в данных!",
                                   plot_filename=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        ts = df.set_index(date_col)[target_col]

        # Загружаем модель вместе с информацией об экзогенных колонках
        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        model = model_package["model"]
        exog_cols = model_package.get("exog_cols", None)

        # Если модель обучалась с экзогенными переменными, требуется загрузить файл с будущими значениями факторов
        if exog_cols:
            exog_forecast_file = request.files.get("exog_forecast_file")
            if not exog_forecast_file:
                return render_template("predict.html",
                                       message="Для прогноза с внешними факторами необходимо загрузить CSV с будущими значениями факторов.",
                                       plot_filename=None, data_files=data_files,
                                       model_files=model_files, script_files=script_files)
            forecast_exog_df = pd.read_csv(exog_forecast_file)
            missing_forecast_cols = [col for col in exog_cols if col not in forecast_exog_df.columns]
            if missing_forecast_cols:
                return render_template("predict.html",
                                       message=f"В файле с прогнозом отсутствуют колонки: {missing_forecast_cols}",
                                       plot_filename=None, data_files=data_files,
                                       model_files=model_files, script_files=script_files)
            # Гарантируем, что порядок колонок соответствует обучению
            forecast_exog_df = forecast_exog_df[exog_cols]
        else:
            forecast_exog_df = None

        # Получаем прогноз через метод get_forecast
        try:
            forecast_result = model.get_forecast(steps=forecast_steps, exog=forecast_exog_df)
            forecast_values = forecast_result.predicted_mean
        except Exception as e:
            return render_template("predict.html",
                                   message=f"Ошибка при прогнозе: {e}",
                                   plot_filename=None, data_files=data_files,
                                   model_files=model_files, script_files=script_files)

        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='D')[1:]
        forecast_series = pd.Series(forecast_values, index=future_dates)

        plot_filename = save_forecast_plot(ts, forecast_series, model_name, current_app.config["PLOT_FOLDER"])

        return render_template("predict.html",
                               message="Прогноз успешно сформирован!",
                               plot_filename=plot_filename,
                               data_files=data_files,
                               model_files=model_files,
                               script_files=script_files,
                               selected_csv=csv_filename,
                               selected_script=script_name,
                               selected_model=model_name,
                               entered_date_col=date_col,
                               entered_target_col=target_col,
                               entered_steps=forecast_steps)

    return render_template("predict.html",
                           message=None, plot_filename=None,
                           data_files=data_files,
                           model_files=model_files,
                           script_files=script_files,
                           selected_csv=None,
                           selected_script=None,
                           selected_model=None,
                           entered_date_col=None,
                           entered_target_col=None,
                           entered_steps=None)
