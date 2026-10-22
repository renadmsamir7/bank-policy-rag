
# Bank Policy Q&A — Local RAG API

An AI-powered banking policy question-answering system built with Python, FastAPI, ChromaDB, and Ollama.

The application uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded banking documents, with references to the original PDF files and page numbers.

## Features

- PDF text extraction using PyPDF and OCR
- Two-column document processing using Apple's Vision OCR
- Text chunking with source metadata
- Semantic search using ChromaDB
- Local embeddings with all-MiniLM-L6-v2
- Answer generation using Qwen2.5 3B through Ollama
- REST API built with FastAPI
- Retrieved document references in API responses
- Automated API tests using pytest

## Technology Stack

- Python 3.11
- FastAPI and Uvicorn
- ChromaDB
- Ollama and Qwen2.5 3B
- PyPDF and PyMuPDF
- ocrmac
- LangChain Text Splitters
- pytest

## How It Works

1. PDF documents are loaded and their text is extracted.
2. Extracted text is divided into smaller chunks.
3. Text chunks are converted into embeddings.
4. Embeddings and source metadata are stored in ChromaDB.
5. User questions are matched with relevant document passages.
6. Qwen generates an answer using the retrieved context.
7. The API returns the answer and retrieved source references.

## Local Setup

The current PDF processing pipeline requires macOS because it uses Apple's Vision OCR.

### 1. Install Python dependencies

Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
```

### 2. Install Ollama

Download Ollama from https://ollama.com/download

Download the language model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before using the API.

### 3. Add PDF documents

Place your PDF files inside:

```text
data/sample_policies/
```

### 4. Process the documents

Run:

```bash
python app/pdf_loader.py
```

This generates `data/processed_chunks.json`.

### 5. Build the vector database

Run:

```bash
python app/index_chunks.py
```

### 6. Start the API

Run:

```bash
python -m uvicorn main:app --app-dir app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoint

### POST /ask

Example request:

```json
{
  "question": "What is the deadline for disputing a card transaction?"
}
```

The API returns:

- The submitted question
- An AI-generated answer
- Retrieved source filenames and page numbers

Retrieved sources are search matches and may include passages not directly used in the final answer.

## Running Tests

```bash
PYTHONPATH=app python -m pytest tests/test_api.py -v
```

The automated tests cover input validation, API responses, simulated Ollama failures, and retrieval efficiency.

## Limitations

- Answers depend on the quality and relevance of the uploaded documents.
- OCR may introduce text recognition errors.
- The model may occasionally generate inaccurate answers or citations.
- The system does not automatically verify whether banking policies are up to date.
- The current API is intended for local development and experimentation.

## Data Privacy

Original banking PDFs, extracted text, and the local ChromaDB database are excluded from version control.

## Project Status

Working local RAG API with document retrieval, answer generation, source references, and automated tests.

Further improvements to evaluation, error handling, and deployment are planned.
