import csv
import io

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader
from pptx import Presentation

from app.services.document_export_service import generate_document

CONTENT = """## Crop comparison

| Crop | Rotation interval | Benefit |
| --- | --- | --- |
| Wheat | 3 years | Breaks disease cycles |
| Peas | 4 years | Adds nitrogen |
"""


def test_generate_pdf():
    document = generate_document("Crop rotation", CONTENT, "pdf")
    assert document.filename == "Crop-rotation.pdf"
    assert document.payload.startswith(b"%PDF")
    assert len(PdfReader(io.BytesIO(document.payload)).pages) == 1


def test_generate_docx():
    document = generate_document("Crop rotation", CONTENT, "docx")
    parsed = Document(io.BytesIO(document.payload))
    assert parsed.paragraphs[0].text == "Crop rotation"
    assert parsed.tables[0].cell(1, 0).text == "Wheat"


def test_generate_csv():
    document = generate_document("Crop rotation", CONTENT, "csv")
    rows = list(csv.reader(io.StringIO(document.payload.decode("utf-8-sig"))))
    assert rows[0] == ["Crop", "Rotation interval", "Benefit"]
    assert rows[2][0] == "Peas"


def test_generate_xlsx():
    document = generate_document("Crop rotation", CONTENT, "xlsx")
    workbook = load_workbook(io.BytesIO(document.payload))
    sheet = workbook.active
    assert sheet["A1"].value == "Crop"
    assert sheet["A3"].value == "Peas"


def test_generate_pptx():
    document = generate_document("Crop rotation", CONTENT, "pptx")
    presentation = Presentation(io.BytesIO(document.payload))
    assert len(presentation.slides) >= 2
    assert presentation.slides[0].shapes.title.text == "Crop rotation"


def test_filename_is_sanitized():
    document = generate_document("../../Unsafe title!", "Body", "csv")
    assert document.filename == "Unsafe-title.csv"
