import os
import pandas as pd
from flask import Blueprint, request, render_template, current_app
from flask_login import current_user

from db.models import Plot, UserAction, db

bp = Blueprint("index", __name__)


@bp.route("/", methods=["GET", "POST"], endpoint="index")
def index():
    # Если пользователь авторизован, показываем дашборд
    if current_user.is_authenticated:
        # Подсчёт файлов моделей (.pkl) в MODEL_FOLDER
        model_files = [f for f in os.listdir(current_app.config['MODEL_FOLDER']) if f.endswith(".pkl")]
        models_count = len(model_files)

        # Подсчёт CSV-файлов в UPLOAD_FOLDER
        csv_files = [f for f in os.listdir(current_app.config['UPLOAD_FOLDER']) if f.endswith(".csv")]
        csv_count = len(csv_files)

        # Количество графиков (считаем все записи в Plot для текущего пользователя)
        forecasts_count = db.session.query(Plot).filter_by(user_id=current_user.id).count()

        # Количество ошибок (предполагаем, что в UserAction записи с action_type="error")
        errors_count = db.session.query(UserAction).filter_by(user_id=current_user.id, action_type="error").count()

        metrics = {
            "models_count": models_count,
            "csv_count": csv_count,
            "forecasts_count": forecasts_count,
            "errors_count": errors_count
        }

        # Получаем последние 5 графиков
        recent_plots = (Plot.query
                        .filter_by(user_id=current_user.id)
                        .order_by(Plot.created_at.desc())
                        .limit(5)
                        .all())

        # Получаем последние 5 действий пользователя
        recent_actions = (UserAction.query
                          .filter_by(user_id=current_user.id)
                          .order_by(UserAction.created_at.desc())
                          .limit(5)
                          .all())

        # Если в POST (например, загрузка CSV) — оставляем исходную логику
        if current_app.request.method == "POST":
            file = current_app.request.files.get("file")
            if file:
                filename = file.filename
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Здесь можно добавить запись о загрузке в UserAction
                action = UserAction(
                    user_id=current_user.id,
                    action_type="upload",
                    description=f"Файл {filename} жүктелді."
                )
                db.session.add(action)
                db.session.commit()
                message = f"Файл '{filename}' жүктелді!"
                # Обновляем csv_count после загрузки файла
                csv_files = [f for f in os.listdir(current_app.config['UPLOAD_FOLDER']) if f.endswith(".csv")]
                metrics["csv_count"] = len(csv_files)
            else:
                message = None
        else:
            message = None

        return render_template("dashboard.html",
                               metrics=metrics,
                               recent_plots=recent_plots,
                               recent_actions=recent_actions,
                               message=message)
    else:
        # Если пользователь не авторизован, показываем лендинговую страницу
        return render_template("lending.html")


@bp.route("/faq")
def faq():
    return render_template("faq.html")

