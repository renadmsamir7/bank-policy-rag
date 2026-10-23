# Bank Policy Q&A — Local RAG API

A local, document-grounded banking policy question-answering API built with Python, FastAPI, ChromaDB, BM25, Ollama, and Docker.

The application uses Retrieval-Augmented Generation (RAG) to answer questions grounded in banking policy documents. It uses hybrid retrieval (semantic similarity + BM25 keyword matching), a locally hosted language model, and source references to answer questions from uploaded banking policy PDFs.

The project demonstrates an end-to-end RAG workflow, including PDF processing, text extraction, vector embeddings, hybrid retrieval and rank fusion, adjacent-chunk context expansion, local answer generation, source citation formatting, REST API development, automated evaluation, and containerization.



## Project Demo 🎬

Watch the Bank Policy RAG API in action!

This demonstration showcases:
- Banking policy question answering with PDF citations.
- Debit card replacement fee retrieval.
- Safe handling of unsupported questions.

https://github.com/user-attachments/assets/876cf63b-1d73-4242-8b93-10df678229f5



## Features

- PDF text extraction with PyPDF and OCR; macOS Apple Vision OCR for two-column PDFs
- Text chunking with original PDF filename and page metadata
- Local embeddings using `all-MiniLM-L6-v2` and persistent storage in ChromaDB
- **Hybrid retrieval:** ChromaDB semantic search + BM25 keyword retrieval, combined with reciprocal rank fusion
- **Neighboring-chunk expansion:** appends the preceding chunk when a retrieved passage begins mid-sentence (same file and page)
- Local answer generation with **Qwen2.5 3B** through Ollama
- Document-grounded prompting and a fixed refusal response for unsupported questions
- Source-aware answers with short citation expansion to PDF filename and page number
- Targeted output cleanup for known merged-word errors (not a general grammar checker)
- FastAPI endpoints for health checks and policy questions; `answer_status` included in responses
- Unit/API tests, retrieval evaluation, and answer evaluation
- Docker Compose for a local API connected to host Ollama, persistent ChromaDB, and processed chunks
- No hosted language-model API required

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
- BM25 (`rank-bm25`)
- Reciprocal rank fusion (RRF)
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
- Python evaluation scripts
- Docker
- Docker Compose

## How It Works

1. PDF files are parsed and, when necessary, processed with OCR. Text is split into chunks with source filename and page metadata.
2. Chunk embeddings are generated with `all-MiniLM-L6-v2` and indexed in persistent ChromaDB. The processed chunks also serve as input for BM25 keyword retrieval.
3. For a user question, the API runs semantic and BM25 retrieval and combines their rankings using reciprocal rank fusion (RRF).
4. The selected passages are formatted with numbered source labels. When a passage appears to start mid-sentence, the context builder can include its preceding chunk from the same PDF page.
5. Qwen2.5 3B, running locally through Ollama, generates a document-grounded response.
6. The API corrects a few known word-spacing errors, expands existing short source citations using retrieved metadata, and assigns `answer_status` (`generated` or `not_confirmed`).
7. FastAPI returns the answer and a structured list of retrieved sources. These are *retrieval matches*, not independently verified evidence for each generated claim.

### RAG Pipeline

```text
Banking policy PDFs
         |
   PDF parsing / OCR
         |
  Chunking + metadata
         |
         +----------------------+
         |                      |
  Embedding + ChromaDB    Processed chunks
   (semantic index)         (BM25 corpus)
         |                      |
         +----------+-----------+
                    |
                User question
                    |
        Semantic + BM25 retrieval
                    |
          Reciprocal rank fusion
                    |
           Selected passages
                    |
        Adjacent-chunk expansion
                    |
             Prompt builder
                    |
           Ollama / Qwen2.5
                    |
        Formatting + citations
                    |
             FastAPI JSON
 (question, answer, answer_status,
           retrieved_sources)
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

PDF parsing and OCR are separate processing steps. For documents requiring OCR, run the project's macOS OCR workflow, then complete text chunking to produce:

```text
data/processed_chunks.json
```

**Check that this file exists before indexing.** The available scripts are not a guaranteed one-command import pipeline for every PDF. Each chunk contains `text`, `file_name`, and `page`.

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
- `data/processed_chunks.json` exists (used to expand neighboring context and run BM25 retrieval).

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
- Mounts `data/processed_chunks.json` read-only for keyword search and neighboring-chunk lookup.
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

**Note:** Both `chroma_db/` and `data/processed_chunks.json` are local inputs. They must exist before starting the API container; neither is provided in this repository.

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

**Example response (illustrative, based on a tested question):**

```json
{
  "question": "What is the deadline for disputing a card transaction?",
  "answer": "The deadline for disputing a card transaction is within thirty days from the sending date of the Statement of Account on which the disputed Transaction appeared. [Source 1] (cb-retail-terms-and-conditions-en.pdf, page 8)",
  "answer_status": "generated",
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

`answer_status` is `generated` for a non-refusal response or `not_confirmed` when the answer is the exact configured refusal message. **`generated` does not mean the answer has been independently fact-checked.**

Retrieved sources are ranked search matches. Only cited passages should be interpreted as the model's claimed support, and citations themselves are not independently validated. The citation formatter expands references such as `[Source 1]` to the corresponding filename and page; it does not generate citations when the model omits them.

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

## Running Tests and Evaluations

### Automated tests

Run the API and formatting regression tests locally:

```bash
PYTHONPATH=app:tests python -m pytest tests/test_api.py tests/test_answer_formatting.py -v
```

The API tests use mocking for external services where appropriate. Formatting tests verify known merged-word detection independently of live model responses.

### Retrieval evaluation

```bash
PYTHONPATH=app python tests/evaluate_retrieval.py
```

The evaluation checks whether selected questions retrieve the expected source filename/page and supporting excerpt. **The current handpicked five-case set previously achieved HitRate@3 = 5/5 (100%) with hybrid retrieval.** This is a small development set, not a general retrieval-quality guarantee.

### Answer evaluation

First start Ollama and the Docker API, then run:

```bash
PYTHONPATH=app python tests/evaluate_answers.py
```

The evaluation compares expected answer terms, expected retrieved source and citation, the API `answer_status`, and a few known formatting errors. **The latest three-case evaluation passed 3/3 (100%)**; this small result should not be interpreted as overall banking-policy accuracy. A passing citation check confirms that the expected source was *referenced*, not that every generated statement is fully supported.

## Project Structure

```text
bank-policy-rag/
├── app/
│   ├── main.py                  # FastAPI + answer post-processing
│   ├── hybrid_retriever.py      # Combine semantic and BM25 results
│   ├── retriever.py             # ChromaDB semantic search
│   ├── keyword_retriever.py     # BM25 keyword search
│   ├── context_builder.py      # Source labels + neighboring chunks
│   ├── prompt_builder.py        # Grounded LLM instructions
│   ├── answer_formatter.py      # Known spacing corrections
│   ├── pdf_loader.py            # PDF extraction
│   └── index_chunks.py         # Vector indexing
├── data/
│   ├── sample_policies/         # Local PDFs (not committed)
│   └── processed_chunks.json   # Local generated chunks (not committed)
├── chroma_db/                  # Local vector DB (not committed)
├── tests/
│   ├── test_api.py
│   ├── test_answer_formatting.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   ├── rag_eval_cases.json
│   └── answer_eval_cases.json
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-api.txt
├── .dockerignore
├── .env.example
├── .gitignore
└── README.md
```

Some generated or local-only directories and files will not appear in a fresh clone.

## Limitations

- Answer quality depends on the relevance and accuracy of the source documents.
- OCR may introduce text recognition errors.
- Even hybrid retrieval can return unrelated or incomplete passages; neighboring-chunk expansion uses a heuristic and does not fix every split.
- The current evaluation suites contain only a small number of handpicked questions.
- The language model may occasionally generate inaccurate statements, omit citations, or cite an irrelevant source.
- Targeted formatting corrections cover only a few observed merged words, not arbitrary language errors.
- The current implementation does not independently verify that generated claims are fully supported by their cited sources.
- Banking policy updates are not detected automatically.
- PDF processing currently relies on macOS-specific OCR functionality for certain documents.
- Both the vector database and processed chunk JSON must be generated before running the Dockerized API.
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

**Functional, local-first RAG API with Docker Compose support.**

Implemented and exercised:

- PDF extraction and OCR-assisted processing with filename/page metadata
- MiniLM embeddings and ChromaDB indexing
- Semantic + BM25 hybrid retrieval with reciprocal rank fusion
- Limited neighboring-chunk expansion to preserve split policy requirements
- Document-grounded Qwen2.5 answer generation via Ollama
- FastAPI endpoints, `answer_status`, and retrieved-source metadata
- Citation expansion and targeted answer formatting
- API/formatting tests and small retrieval/answer evaluation sets
- Docker Compose setup with persistent vector storage and read-only processed chunks

**Latest recorded checks:** 3/3 answer evaluation cases passed, including answer formatting and citations; five selected retrieval cases achieved HitRate@3 = 5/5 with hybrid retrieval. These are small development checks, not claims of production readiness.

### Potential Future Improvements

- Expand evaluation with varied, adversarial, and ambiguous questions
- Test citation faithfulness and factual support at the claim level
- Improve OCR quality, chunk boundaries, and handling of conflicting policy rules
- Add broader integration tests and continuous integration (CI)
- Add authentication, rate limits, logging, and security controls before deployment
- Support more document formats, corpus updates, and production-oriented deployment

## Disclaimer

This project is intended for educational, portfolio, and software development purposes.

AI-generated answers should not be treated as official banking advice. Users should consult the original policy documents or the relevant financial institution for authoritative information.
