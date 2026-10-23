
from io import BytesIO
from unittest.mock import patch
from urllib.error import URLError

from fastapi.testclient import TestClient
from main import app


# ==========================================
# STEP 1: Create our FastAPI test client
# ==========================================

client = TestClient(app)


# ==========================================
# TEST 1: Reject empty questions
# ==========================================

def test_empty_question():

    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Question cannot be empty"
    }


# ==========================================
# TEST 2: Verify the home endpoint
# ==========================================

def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Bank Policy RAG API is running"
    }


# ==========================================
# TEST 3: Reject whitespace-only questions
# ==========================================

def test_whitespace_question():

    response = client.post(
        "/ask",
        json={"question": "     "}
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Question cannot be empty"
    }


# ==========================================
# TEST 4: Verify AI answer and source metadata
# ==========================================

def test_ask_returns_answer_and_sources():

    # Simulate Qwen's response.
    fake_llm_response = BytesIO(
        b'{"response": "The deadline is 30 days."}'
    )

    # Simulate a retrieved banking passage.
    fake_passages = [
        {
            "text": "Disputes must be submitted within thirty days.",
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 8,
            "distance": 0.78
        }
    ]

    # Mock prompt building, LLM call, and retrieval.
    with (
        patch("main.build_prompt", return_value="Test prompt"),
        patch("main.urlopen", return_value=fake_llm_response),
        patch("main.retrieve_passages", return_value=fake_passages)
    ):

        response = client.post(
            "/ask",
            json={
                "question": "What is the card dispute deadline?"
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "What is the card dispute deadline?"
    )

    assert data["answer"] == "The deadline is 30 days."

    assert data["answer_status"] == "generated"

    assert len(data["retrieved_sources"]) == 1

    source = data["retrieved_sources"][0]

    assert source["rank"] == 1
    assert source["page"] == 8

    assert source["file_name"] == (
        "cb-retail-terms-and-conditions-en.pdf"
    )


# ==========================================
# TEST 5: Handle Ollama connection failures
# ==========================================

def test_ollama_unavailable():

    # Simulate Ollama being offline.
    with (
        patch("main.build_prompt", return_value="Test prompt"),
        patch(
            "main.urlopen",
            side_effect=URLError("Connection refused")
        )
    ):

        response = client.post(
            "/ask",
            json={
                "question": "What are the bank account fees?"
            }
        )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Ollama is unavailable. Check that it is running."
    }


# ==========================================
# TEST 6: Ensure retrieval happens only once
# ==========================================

def test_retrieval_happens_only_once():

    question = "What is the card dispute deadline?"

    fake_passages = [
        {
            "text": "Disputes must be submitted within thirty days.",
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 8,
            "distance": 0.78
        }
    ]

    fake_llm_response = BytesIO(
        b'{"response": "The deadline is 30 days."}'
    )

    # Mock retrieval, prompt building, and Ollama.
    with (
        patch(
            "main.retrieve_passages",
            return_value=fake_passages
        ) as mock_retrieve,

        patch(
            "main.build_prompt",
            return_value="Test prompt"
        ) as mock_build_prompt,

        patch(
            "main.urlopen",
            return_value=fake_llm_response
        )
    ):

        response = client.post(
            "/ask",
            json={"question": question}
        )

    assert response.status_code == 200

    # Retrieval should happen exactly once.
    mock_retrieve.assert_called_once_with(
        question,
        n_results=3
    )

    # Prompt building should reuse the same passages.
    mock_build_prompt.assert_called_once_with(
        question,
        passages=fake_passages
    )


# ==========================================
# TEST 7: Verify API health check
# ==========================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "Bank Policy RAG API"
    }


# ==========================================
# TEST 8: Verify unsupported answer status
# ==========================================

def test_unsupported_answer_status():

    # Simulate Qwen declining to answer.
    fake_llm_response = BytesIO(
        b'{"response": "I cannot confirm the answer from the provided documents."}'
    )

    fake_passages = [
        {
            "text": "General banking terms and conditions.",
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 6,
            "distance": 0.85
        }
    ]

    with (
        patch("main.build_prompt", return_value="Test prompt"),
        patch("main.urlopen", return_value=fake_llm_response),
        patch("main.retrieve_passages", return_value=fake_passages)
    ):

        response = client.post(
            "/ask",
            json={
                "question": (
                    "How many days of paid annual leave "
                    "do bank employees receive?"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "I cannot confirm the answer from the provided documents."
    )

    assert data["answer_status"] == "not_confirmed"


# ==========================================
# TEST 9: Verify answer formatting and citations
# ==========================================

def test_api_formats_answer_and_citations():

    # Simulate a Qwen answer containing a merged-word
    # error and an incomplete source citation.
    fake_llm_response = BytesIO(
        b'{"response": "The statement on whichthe disputed Transaction appeared. [Source 1]"}'
    )

    # Simulate the retrieved banking passage.
    fake_passages = [
        {
            "text": (
                "The disputed Transaction appeared "
                "on the Statement of Account."
            ),
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 8,
            "distance": 0.78
        }
    ]

    # Mock retrieval, prompt building, and Ollama.
    with (
        patch(
            "main.build_prompt",
            return_value="Test prompt"
        ),
        patch(
            "main.urlopen",
            return_value=fake_llm_response
        ),
        patch(
            "main.retrieve_passages",
            return_value=fake_passages
        )
    ):

        response = client.post(
            "/ask",
            json={
                "question": "What is the card dispute deadline?"
            }
        )

    # Verify successful API response.
    assert response.status_code == 200

    data = response.json()

    # Verify the formatting error is corrected.
    assert "whichthe" not in data["answer"].casefold()

    assert "which the" in data["answer"].casefold()

    # Verify the complete PDF citation is added.
    expected_citation = (
        "[Source 1] "
        "(cb-retail-terms-and-conditions-en.pdf, page 8)"
    )

    assert expected_citation in data["answer"]

    # Verify the complete expected answer.
    expected_answer = (
        "The statement on which the disputed "
        "Transaction appeared. "
        "[Source 1] "
        "(cb-retail-terms-and-conditions-en.pdf, page 8)"
    )

    assert data["answer"] == expected_answer

    # Verify the answer status is preserved.
    assert data["answer_status"] == "generated"

    # Verify source metadata has not changed.
    assert data["retrieved_sources"] == [
        {
            "rank": 1,
            "file_name": "cb-retail-terms-and-conditions-en.pdf",
            "page": 8
        }
    ]
