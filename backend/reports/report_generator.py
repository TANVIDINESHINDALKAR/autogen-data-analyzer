# backend/reports/report_generator.py

import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_pdf_report(
    analysis_text: str,
    output_dir: str,
    report_id: str
):
    """
    Generates a PDF report from analysis text.
    Runs in background (non-blocking).
    """

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{report_id}.pdf")

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    text_obj = c.beginText(40, height - 50)
    text_obj.setFont("Helvetica", 10)

    for line in analysis_text.split("\n"):
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()

    return file_path
