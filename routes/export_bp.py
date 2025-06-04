import os
import io
import pandas as pd
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    current_app,
    send_file,
    flash
)
from flask_login import login_required
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

export_bp = Blueprint(
    "export",
    __name__,
    url_prefix="/export",
    template_folder="templates"
)

@export_bp.route("/export_page", methods=["GET"], endpoint="export_page")
@login_required
def export_page():
    """
    Страница со списком всех CSV из UPLOAD_FOLDER
    и кнопками для конвертации в Excel / PDF.
    """
    data_dir = current_app.config["UPLOAD_FOLDER"]
    try:
        all_files = [
            f for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f)) and f.lower().endswith(".csv")
        ]
    except FileNotFoundError:
        all_files = []
        flash("data папка табылмады.", "error")

    query = request.args.get("query", "").lower()
    if query:
        data_files = [f for f in all_files if query in f.lower()]
    else:
        data_files = all_files

    return render_template("export_page.html", data_files=data_files, query=query)


@export_bp.route("/export/excel/<filename>", methods=["GET"], endpoint="export_excel")
@login_required
def export_excel(filename):
    """
    Конвертация выбранного CSV → Excel (.xlsx)
    """
    if "/" in filename or "\\" in filename or not filename.lower().endswith(".csv"):
        flash("CSV файлының атын дұрыстап көрсетіңіз.", "error")
        return redirect(url_for("export.export_page"))

    data_dir = current_app.config["UPLOAD_FOLDER"]
    csv_path = os.path.join(data_dir, filename)
    if not os.path.isfile(csv_path):
        flash(f"CSV файл «{filename}» табылмады.", "error")
        return redirect(url_for("export.export_page"))

    try:
        df = pd.read_csv(csv_path)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        output.seek(0)

        xls_name = filename[:-4] + ".xlsx"
        return send_file(
            output,
            as_attachment=True,
            download_name=xls_name,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        flash(f"Excel жасап шыққанда қате: {e}", "error")
        return redirect(url_for("export.export_page"))


@export_bp.route("/export/pdf/<filename>", methods=["GET"], endpoint="export_pdf")
@login_required
def export_pdf(filename):
    """
    Конвертация выбранного CSV → PDF
    """
    if "/" in filename or "\\" in filename or not filename.lower().endswith(".csv"):
        flash("CSV файлының атын дұрыстап көрсетіңіз.", "error")
        return redirect(url_for("export.export_page"))

    data_dir = current_app.config["UPLOAD_FOLDER"]
    csv_path = os.path.join(data_dir, filename)
    if not os.path.isfile(csv_path):
        flash(f"CSV файл «{filename}» табылмады.", "error")
        return redirect(url_for("export.export_page"))

    try:
        df = pd.read_csv(csv_path)
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Заголовок
        title = Paragraph(f"Тізім: {filename}", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Строим таблицу
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data, repeatRows=1)
        tbl_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ])
        table.setStyle(tbl_style)

        for i in range(1, len(data)):
            bg = colors.whitesmoke if i % 2 == 0 else colors.lightgrey
            table.setStyle(TableStyle([("BACKGROUND", (0, i), (-1, i), bg)]))

        elements.append(table)
        doc.build(elements)
        pdf_buffer.seek(0)

        pdf_name = filename[:-4] + ".pdf"
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=pdf_name,
            mimetype="application/pdf"
        )
    except Exception as e:
        flash(f"PDF жасап шыққанда қате: {e}", "error")
        return redirect(url_for("export.export_page"))
