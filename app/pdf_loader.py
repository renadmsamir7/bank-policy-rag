
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from pypdf import PdfReader

POLICIES_FOLDER = Path("data/sample_policies")
pdf_files = list(POLICIES_FOLDER.glob("*.pdf"))

documents = []

for pdf_path in pdf_files:
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

if documents:
    print("\nFirst document:")
    print(f"File: {documents[0]['file_name']}")
    print(f"Page: {documents[0]['page']}")
    print(f"Characters: {len(documents[0]['text'])}")

# Create our first text chunk
first_document = documents[0]

first_chunk = {
    "text": first_document["text"][:500],
    "file_name": first_document["file_name"],
    "page": first_document["page"]
}

print("\nFirst chunk:")
print(first_chunk)


# Create our text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Store all generated chunks
chunks = []

# Process each extracted PDF page
for document in documents:

    # Skip empty pages
    if not document["text"].strip():
        continue

    # Split the page into smaller pieces
    page_chunks = text_splitter.split_text(document["text"])

    # Save every chunk with its metadata
    for chunk_text in page_chunks:
        chunks.append({
            "text": chunk_text,
            "file_name": document["file_name"],
            "page": document["page"]
        })

print(f"\nTotal chunks created: {len(chunks)}")

# Display the first chunk
if chunks:
    print("\nFirst automatic chunk:")
    print(chunks[0])

