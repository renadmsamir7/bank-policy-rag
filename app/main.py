
import os
import json
import re

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prompt_builder import build_prompt
from hybrid_retriever import hybrid_search as retrieve_passages
from answer_formatter import format_answer_text


# ==========================================
# STEP 1: Load environment configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env")

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
# STEP 4: Format source citations
# ==========================================

def format_source_citations(answer, retrieved_sources):
    """
    Expand short source references into complete citations.

    Example:
    [Source 1]

    Becomes:
    [Source 1] (filename.pdf, page 8)

    Only sources present in the retrieved results can
    be expanded.

    Existing full citations are left unchanged.
    """

    # Create a lookup table using source rank.
    source_lookup = {
        str(source["rank"]): source
        for source in retrieved_sources
    }

    def replace_citation(match):

        rank = match.group(1)

        source = source_lookup.get(rank)

        # Do not invent metadata for an unknown source.
        if source is None:
            return match.group(0)

        file_name = source["file_name"]
        page = source["page"]

        return (
            f"[Source {rank}] "
            f"({file_name}, page {page})"
        )

    # Expand short references such as [Source 1].
    # Avoid duplicating existing full citations.
    formatted_answer = re.sub(
        r"\[Source\s+(\d+)\](?!\s*\()",
        replace_citation,
        answer,
        flags=re.IGNORECASE
    )

    return formatted_answer


# ==========================================
# STEP 5: Home endpoint
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Bank Policy RAG API is running"
    }


# ==========================================
# STEP 6: Health check endpoint
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Bank Policy RAG API"
    }


# ==========================================
# STEP 7: Banking question-answering endpoint
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    # 1. Validate the question.
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # 2. Retrieve relevant banking passages once.
    passages = retrieve_passages(
        question,
        n_results=3
    )

    # 3. Build the prompt using the retrieved passages.
    prompt = build_prompt(
        question,
        passages=passages
    )

    # 4. Prepare the request for Qwen.
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

    # 5. Generate the AI answer.
    try:
        with urlopen(ollama_request, timeout=300) as response:
            result = json.load(response)

    except (URLError, TimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Ollama is unavailable. Check that it is running."
        )

    # 6. Format retrieved sources as structured data.
    retrieved_sources = [
        {
            "rank": i,
            "file_name": passage["source"],
            "page": passage["page"]
        }
        for i, passage in enumerate(passages, start=1)
    ]

    # ==========================================
    # STEP 8: Process the generated answer
    # ==========================================

    # 7. Extract the model's answer.
    answer = result["response"].strip()

    # 8. NEW: Correct known merged-word errors.
    answer = format_answer_text(answer)

    # 9. Expand short citations using retrieved metadata.
    answer = format_source_citations(
        answer,
        retrieved_sources
    )

    # ==========================================
    # STEP 9: Determine answer status
    # ==========================================

    refusal_message = (
        "I cannot confirm the answer from the provided documents."
    )

    if answer.casefold() == refusal_message.casefold():
        answer_status = "not_confirmed"
    else:
        answer_status = "generated"

    # ==========================================
    # STEP 10: Return the API response
    # ==========================================

    return {
        "question": question,
        "answer": answer,
        "answer_status": answer_status,
        "retrieved_sources": retrieved_sources
    }
