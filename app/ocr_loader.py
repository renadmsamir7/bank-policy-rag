
import pymupdf
from PIL import Image
from ocrmac import ocrmac


def extract_retail_page_text(page):
    """Extract text from a two-column PDF page."""

    width = page.rect.width
    height = page.rect.height

    columns = [
        pymupdf.Rect(0, 0, width / 2, height),
        pymupdf.Rect(width / 2, 0, width, height)
    ]

    extracted_columns = []

    for area in columns:
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            clip=area,
            colorspace=pymupdf.csRGB,
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples
        )

        ocr = ocrmac.OCR(
            image,
            recognition_level="accurate",
            language_preference=["en-US"]
        )

        results = ocr.recognize()

        column_text = "\n".join(
            text for text, confidence, box in results
        )

        extracted_columns.append(column_text)

    return "\n".join(extracted_columns)
