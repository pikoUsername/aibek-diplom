import os

import pdfkit
import datetime

from flask import Blueprint, request, current_app, render_template, send_file

bp = Blueprint("reports", __name__)


@bp.route("/generate_report", methods=["POST"])
def generate_report():
    model_name = request.form.get("model_name")
    # Допустим, мы уже посчитали метрики:
    mse = 10
    rmse = 3.162
    mae = 2.5
    mape = 5.2
    plot_path = os.path.join(current_app.config["PLOT_FOLDER"], "forecast_example.png")

    # Читаем параметры модели (из pickle, например)
    model_params = "(p,d,q)=(1,1,1)"

    # Рендерим HTML через render_template
    html_content = render_template("report_template.html",
                                   generation_date=datetime.date.today().strftime("%d.%m.%Y"),
                                   model_name=model_name,
                                   model_params=model_params,
                                   mse=mse,
                                   rmse=rmse,
                                   mae=mae,
                                   mape=mape,
                                   plot_path=plot_path)

    # Генерируем PDF (pdfkit.from_string or from_file)
    pdf_output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{model_name}_report.pdf")
    pdfkit.from_string(html_content, pdf_output_path)

    return send_file(pdf_output_path, as_attachment=True)


@bp.route("/generate_excel", methods=["POST"])
def generate_excel():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Model"
    sheet["B1"] = "MSE"
    # ... заполняем данные
    excel_path = os.path.join(current_app.config["UPLOAD_FOLDER"], "report.xlsx")
    workbook.save(excel_path)
    return send_file(excel_path, as_attachment=True)

