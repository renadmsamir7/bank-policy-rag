
# Bank Policy Q&A — Local RAG API

An AI-powered banking policy question-answering system built with Python, FastAPI, ChromaDB, Ollama, and Docker.

The application uses Retrieval-Augmented Generation (RAG) to answer questions grounded in banking policy documents. It combines semantic document retrieval with a locally hosted large language model (LLM) and provides references to the original PDF files and page numbers.

The project demonstrates an end-to-end RAG workflow, including PDF processing, text extraction, vector embeddings, semantic search, answer generation, REST API development, automated testing, and containerization.

## Features

- PDF text extraction using PyPDF and OCR
- Two-column document processing using Apple's Vision OCR
- Text chunking with source metadata
- Semantic search using ChromaDB
- Local text embeddings with all-MiniLM-L6-v2
- Answer generation using Qwen2.5 3B through Ollama
- Context-aware prompting to ground answers in retrieved passages
- Handling of questions that cannot be answered from the documents
- REST API developed with FastAPI
- Source filenames and page numbers included in API responses
- Automated API testing using pytest
- Docker containerization for the API
- Docker Compose for simplified application setup and execution
- Local processing without requiring a cloud-based LLM API

## Technology Stack

**Programming Language**
- Python 3.11

**Backend and API**
- FastAPI
- Uvicorn
- Pydantic

**AI and Retrieval**
- Retrieval-Augmented Generation (RAG)
- ChromaDB
- all-MiniLM-L6-v2 embeddings
- Ollama
- Qwen2.5 3B
- LangChain Text Splitters

**Document Processing**
- PyPDF
- PyMuPDF
- ocrmac
- Apple Vision OCR

**Testing and Containerization**
- pytest
- Docker
- Docker Compose

## How It Works

The application follows a Retrieval-Augmented Generation pipeline:

1. Banking policy PDFs are loaded and their text is extracted using PDF parsing and OCR when needed.
2. Extracted content is divided into smaller text chunks.
3. Each chunk retains metadata identifying its source document and page number.
4. Chunks are converted into vector embeddings using all-MiniLM-L6-v2.
5. Embeddings and metadata are stored in a persistent ChromaDB collection.
6. User questions are converted into embeddings and matched with relevant document passages.
7. Retrieved passages are incorporated into a structured prompt.
8. Qwen2.5 3B generates an answer using the retrieved context.
9. FastAPI returns the generated answer along with the retrieved source references.

### RAG Pipeline

```text
Banking Policy PDFs
        |
        v
PDF Parsing / OCR
        |
        v
Text Chunking + Metadata
        |
        v
Embedding Generation
        |
        v
ChromaDB Vector Database
        |
        v
User Question
        |
        v
Semantic Retrieval
        |
        v
Prompt Construction
        |
        v
Ollama + Qwen2.5 3B
        |
        v
FastAPI Response
(Answer + Retrieved Sources)
```

## Local Setup

The document-processing pipeline currently requires macOS because it uses Apple's Vision OCR.

The API can run locally with Python or inside Docker after the document collection has been processed and indexed.

### 1. Clone the Repository

```bash
git clone https://github.com/renadmsamir7/bank-policy-rag.git
cd bank-policy-rag
```

### 2. Create a Python Virtual Environment

Create a virtual environment:

```bash
python3.11 -m venv .venv
```

Activate it on macOS:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Install Ollama

Download Ollama from:

https://ollama.com/download

Download the Qwen2.5 3B language model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before making API requests.

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

The application uses the following Ollama configuration:

```env
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_API_URL=http://localhost:11434/api/generate
```

Adjust the configuration if Ollama runs at a different address.

### 5. Add Banking Policy Documents

Place the banking policy PDF files inside:

```text
data/sample_policies/
```

The project has been developed and tested using publicly available banking policy documents.

Original PDF files are not included in the GitHub repository.

### 6. Extract and Process Documents

Run:

```bash
python app/pdf_loader.py
```

The document-processing pipeline generates:

```text
data/processed_chunks.json
```

For PDFs requiring OCR, use the project's macOS OCR processing workflow to extract the document text before indexing.

Each processed chunk includes text and source metadata.

### 7. Build the Vector Database

Run:

```bash
python app/index_chunks.py
```

This generates embeddings and stores the processed banking content in a local ChromaDB collection.

The database is saved inside:

```text
chroma_db/
```

The database is excluded from version control and must be created locally.

### 8. Start the API Locally

Run:

```bash
python -m uvicorn main:app --app-dir app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## Running with Docker Compose

The FastAPI application can also run inside Docker using Docker Compose.

Docker runs the API application and connects it to the local ChromaDB database, while Ollama runs separately on the host machine.

The Docker configuration allows the API to run in a Linux container without requiring the macOS-only OCR dependencies.

### Prerequisites

Before starting the application with Docker Compose, ensure that:

- Docker Desktop is installed and running.
- Ollama is installed and running on the host machine.
- The Qwen2.5 3B model has been downloaded.
- Banking documents have already been processed.
- The ChromaDB vector database has already been created.

**Important:** The current Docker configuration runs the RAG API only. PDF extraction, OCR processing, and initial database indexing must be performed separately using the local setup workflow.

### 1. Start the Application

From the project root directory, run:

```bash
docker compose up -d --build
```

This command:

- Builds the FastAPI Docker image.
- Creates and starts the API container.
- Mounts the existing local ChromaDB database.
- Configures communication with Ollama on the host machine.
- Preserves the embedding-model cache using a Docker volume.
- Exposes the API on port 8001.

The `-d` option runs the container in detached mode.

### 2. Verify the Container

Check that the container is running:

```bash
docker compose ps
```

The API should be accessible through:

```text
http://127.0.0.1:8001
```

### 3. Test the Health Endpoint

Run:

```bash
curl http://127.0.0.1:8001/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "Bank Policy RAG API"
}
```

### 4. Access the API Documentation

Open:

http://127.0.0.1:8001/docs

Use the interactive Swagger UI to test the available API endpoints.

### 5. View Container Logs

```bash
docker compose logs -f api
```

### 6. Stop the Application

```bash
docker compose down
```

This stops and removes the Compose container and network without deleting the local ChromaDB database or the persistent model-cache volume.

### Docker Configuration Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Defines the FastAPI container image |
| `requirements-api.txt` | Specifies Linux-compatible API dependencies |
| `compose.yaml` | Configures the API service, environment, networking, and volumes |
| `.dockerignore` | Excludes unnecessary and private files from the Docker build context |

### Docker Networking

The API is exposed locally at:

```text
127.0.0.1:8001
```

The Docker container communicates with Ollama on the host machine through:

```text
http://host.docker.internal:11434/api/generate
```

Ollama is not included in the Docker Compose services and must be running separately.

### Troubleshooting

If port 8001 is already in use, inspect running Docker containers:

```bash
docker ps
```

Stop any older manually started API container that is using the same port.

If Docker Compose has created a container with incorrect networking settings, recreate it using:

```bash
docker compose down
docker compose up -d --build
```

## API Endpoints

### GET /

Provides the application's root endpoint.

### GET /health

Returns the API health status.

Example response:

```json
{
  "status": "ok",
  "service": "Bank Policy RAG API"
}
```

### POST /ask

Accepts a banking policy question and returns an AI-generated answer with retrieved document references.

**Example request:**

```json
{
  "question": "What is the deadline for disputing a card transaction?"
}
```

**Example response:**

```json
{
  "question": "What is the deadline for disputing a card transaction?",
  "answer": "The deadline for disputing a card transaction is within thirty days from the sending date of the Statement of Account on which the disputed Transaction appeared. This information is sourced from [Source 1], File: cb-retail-terms-and-conditions-en.pdf, Page: 8.",
  "retrieved_sources": [
    {
      "rank": 1,
      "file_name": "cb-retail-terms-and-conditions-en.pdf",
      "page": 8
    },
    {
      "rank": 2,
      "file_name": "cb-retail-terms-and-conditions-en.pdf",
      "page": 25
    },
    {
      "rank": 3,
      "file_name": "cb-retail-terms-and-conditions-en.pdf",
      "page": 21
    }
  ]
}
```

The response includes:

- The original question
- The generated answer
- Retrieved document filenames
- Page numbers
- Retrieval rankings

**Note:** Retrieved sources represent semantic search matches. They are not necessarily all used to support the generated answer.

### Handling Unsupported Questions

The assistant is instructed to avoid inventing answers when the retrieved documents do not contain sufficient information.

For example:

**Question:**

```text
How many days of paid annual leave do bank employees receive?
```

**Answer:**

```text
I cannot confirm the answer from the provided documents.
```

This behavior helps reduce unsupported answers, although it does not eliminate the possibility of incorrect model outputs.

## Running Tests

The project includes automated API tests using pytest.

Run:

```bash
PYTHONPATH=app python -m pytest tests/test_api.py -v
```

The tests cover:

- Valid API requests and response formatting
- Input validation
- Health endpoint behavior
- Root endpoint behavior
- Simulated Ollama connection failures
- Retrieval execution and efficiency

Tests use mocking where appropriate to avoid depending on live model responses.

## Project Structure

Key files and directories include:

```text
bank-policy-rag/
|
|-- app/
|   |-- main.py
|   |-- pdf_loader.py
|   |-- index_chunks.py
|   |-- prompt_builder.py
|   |-- context_builder.py
|   `-- ...
|
|-- data/
|   |-- sample_policies/
|   `-- processed_chunks.json
|
|-- chroma_db/
|
|-- tests/
|   `-- test_api.py
|
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- compose.yaml
|-- requirements.txt
|-- requirements-api.txt
`-- README.md
```

The `data/` and `chroma_db/` contents are generated or prepared locally and are not distributed with the repository.

## Limitations

- Answer quality depends on the relevance and accuracy of the source documents.
- OCR may introduce text recognition errors.
- Semantic retrieval may return passages that are not relevant to the final answer.
- The language model may occasionally generate inaccurate statements or citations.
- The current implementation does not independently verify that generated claims are fully supported by their cited sources.
- Banking policy updates are not detected automatically.
- PDF processing currently relies on macOS-specific OCR functionality for certain documents.
- The vector database must be generated before running the Dockerized API.
- Ollama must be available separately on the host machine.
- The API is designed for local development and demonstration rather than unrestricted public deployment.

## Data Privacy

The project is designed to run locally.

Document embeddings and retrieved passages are processed using local components without requiring an external cloud-based language model API.

The following are excluded from version control:

- Original banking PDF files
- Extracted and processed document content
- Local ChromaDB database files
- Local environment configuration containing potentially sensitive settings

Users should review document permissions and data-handling requirements before processing confidential banking information.

## Project Status

**Functional RAG API with Docker Compose support.**

Completed functionality includes:

- Banking PDF text extraction and OCR processing
- Document chunking with source metadata
- Embedding generation and ChromaDB indexing
- Semantic document retrieval
- Context-aware prompt generation
- Local Qwen2.5 answer generation through Ollama
- Retrieval source references in API responses
- Handling of unsupported questions
- REST API implementation with FastAPI
- Automated API testing
- Docker image configuration
- Docker Compose setup and local API verification

The full RAG pipeline has been tested successfully both locally and through Docker Compose.

### Potential Future Improvements

- Automated evaluation of retrieval and answer quality
- Improved relevance filtering and source attribution
- Stronger verification of generated answers against source passages
- Additional integration tests
- API authentication and security enhancements
- Support for more document formats
- Production deployment configuration

## Disclaimer

This project is intended for educational, portfolio, and software development purposes.

AI-generated answers should not be treated as official banking advice. Users should consult the original policy documents or the relevant financial institution for authoritative information.
