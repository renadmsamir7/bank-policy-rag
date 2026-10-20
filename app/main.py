
import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prompt_builder import build_prompt
from retriever import retrieve_passages


# ==========================================
# STEP 1: Create our FastAPI application
# ==========================================

app = FastAPI(
    title="Bank Policy RAG API",
    description="AI-powered banking policy question-answering system",
    version="1.0.0"
)


# ==========================================
# STEP 2: Define the incoming question format
# ==========================================

class QuestionRequest(BaseModel):
    question: str


# ==========================================
# STEP 3: Home endpoint
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Bank Policy RAG API is running"
    }


# ==========================================
# STEP 4: Banking question-answering endpoint
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

    # 2. Retrieve banking information and build the prompt
    prompt = build_prompt(question)

    # 3. Prepare the request for Qwen
    data = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    ollama_request = Request(
        url="http://localhost:11434/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # 4. Generate the AI answer
    try:
        with urlopen(ollama_request, timeout=300) as response:
            result = json.load(response)

    except (URLError, TimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Ollama is unavailable. Check that it is running."
        )

    # 5. Retrieve source information
    passages = retrieve_passages(question, n_results=3)

    # 6. Format retrieved sources as structured data
    retrieved_sources = [
        {
            "rank": i,
            "file_name": passage["source"],
            "page": passage["page"]
        }
        for i, passage in enumerate(passages, start=1)
    ]

    # 7. Return the answer and source information
    return {
        "question": question,
        "answer": result["response"].strip(),
        "retrieved_sources": retrieved_sources
    }
