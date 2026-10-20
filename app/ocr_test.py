
import pymupdf
from ocr_loader import extract_retail_page_text

pdf_path = "data/sample_policies/cb-retail-terms-and-conditions-en.pdf"

# Open the PDF
with pymupdf.open(pdf_path) as pdf:

    # Page 8 (Python index 7)
    page = pdf[7]

    # Extract text using our OCR function
    text = extract_retail_page_text(page)

print("Page: 8")
print("Characters extracted:", len(text))

# Search for the card dispute section
keyword = "disputed transaction"
position = text.lower().find(keyword)

if position != -1:
    print("\nCard dispute section:")
    print(text[max(0, position - 150):position + 900])
else:
    print("\nKeyword not found. Last 1600 characters:")
    print(text[-1600:])
