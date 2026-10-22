
import os
import json

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prompt_builder import build_prompt
from hybrid_retriever import hybrid_search as retrieve_passages

# ==========================================
# STEP 1: Load environment configuration
# ==========================================

# Locate the .env file in our project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Ollama configuration
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

OLLAMA_API_URL = os.getenv(
    "OLLAMA_API_URL",
    "http://localhost:11434/api/generate"
)


# ==========================================
# STEP 2: Create our FastAPI application
# ==========================================

app = FastAPI(
    title="Bank Policy RAG API",
    description="AI-powered banking policy question-answering system",
    version="1.0.0"
)


# ==========================================
# STEP 3: Define the incoming question format
# ==========================================

class QuestionRequest(BaseModel):
    question: str


# ==========================================
# STEP 4: Home endpoint
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Bank Policy RAG API is running"
    }


# ==========================================
# STEP 5: Health check endpoint
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Bank Policy RAG API"
    }


# ==========================================
# STEP 6: Banking question-answering endpoint
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    # 1. Validate the question
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # 2. Retrieve banking passages only once
    passages = retrieve_passages(
        question,
        n_results=3
    )

    # 3. Build the prompt using retrieved passages
    prompt = build_prompt(
        question,
        passages=passages
    )

    # 4. Prepare the request for Qwen
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    ollama_request = Request(
        url=OLLAMA_API_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # 5. Generate the AI answer
    try:
        with urlopen(ollama_request, timeout=300) as response:
            result = json.load(response)

    except (URLError, TimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Ollama is unavailable. Check that it is running."
        )

    # 6. Format retrieved sources as structured data
    retrieved_sources = [
        {
            "rank": i,
            "file_name": passage["source"],
            "page": passage["page"]
        }
        for i, passage in enumerate(passages, start=1)
    ]


    # 7. Identify whether the model declined to answer
    answer = result["response"].strip()

    refusal_message = (
        "I cannot confirm the answer from the provided documents."
    )

    if answer.casefold() == refusal_message.casefold():
        answer_status = "not_confirmed"
    else:
        answer_status = "generated"

    # 8. Return the answer, status, and retrieval matches
    return {
        "question": question,
        "answer": answer,
        "answer_status": answer_status,
        "retrieved_sources": retrieved_sources
    }
