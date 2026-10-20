

import json
import pymupdf

from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ocr_loader import extract_retail_page_text



# ==========================================
# STEP 1: Locate our banking PDFs
# ==========================================

POLICIES_FOLDER = Path("data/sample_policies")
pdf_files = list(POLICIES_FOLDER.glob("*.pdf"))

documents = []


# ==========================================
# STEP 2: Extract text from every PDF page
# ==========================================


for pdf_path in pdf_files:

    # Retail Terms: use OCR for all pages
    if pdf_path.name == "cb-retail-terms-and-conditions-en.pdf":

        with pymupdf.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                print(f"OCR processing page {page_number}...")

                text = extract_retail_page_text(page)

                documents.append({
                    "file_name": pdf_path.name,
                    "page": page_number,
                    "text": text
                })

    # Other PDFs: use normal text extraction
    else:
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            documents.append({
                "file_name": pdf_path.name,
                "page": page_number,
                "text": text
            })



print(f"PDF files found: {len(pdf_files)}")
print(f"Total pages extracted: {len(documents)}")


# ==========================================
# STEP 3: Display the first document
# ==========================================

if documents:
    print("\nFirst document:")
    print(f"File: {documents[0]['file_name']}")
    print(f"Page: {documents[0]['page']}")
    print(f"Characters: {len(documents[0]['text'])}")

    # Create our first manual text chunk
    first_document = documents[0]

    first_chunk = {
        "text": first_document["text"][:500],
        "file_name": first_document["file_name"],
        "page": first_document["page"]
    }

    print("\nFirst chunk:")
    print(first_chunk)


# ==========================================
# STEP 4: Automatically split text into chunks
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []

for document in documents:

    # Skip empty pages
    if not document["text"].strip():
        continue

    # Split the page into chunks
    page_chunks = text_splitter.split_text(document["text"])

    # Store each chunk with its metadata
    for chunk_text in page_chunks:
        chunks.append({
            "text": chunk_text,
            "file_name": document["file_name"],
            "page": document["page"]
        })


print(f"\nTotal chunks created: {len(chunks)}")

if chunks:
    print("\nFirst automatic chunk:")
    print(chunks[0])


# ==========================================
# STEP 5: Verify OCR extraction on page 4
# ==========================================

for document in documents:
    if (
        document["file_name"] == "cb-retail-terms-and-conditions-en.pdf"
        and document["page"] == 4
    ):
        print("\nRetail Terms - Page 4:")
        print(f"Characters extracted: {len(document['text'])}")
        print(document["text"][:1000])
        break


# Save processed chunks for future use
OUTPUT_FILE = Path("data/processed_chunks.json")

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print(f"\nSaved {len(chunks)} chunks to {OUTPUT_FILE}")

