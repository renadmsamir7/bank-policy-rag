
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app

from urllib.error import URLError

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

    # Simulate Qwen's response
    fake_llm_response = BytesIO(
        b'{"response": "The deadline is 30 days."}'
    )

    # Simulate a retrieved banking passage
    fake_passages = [
        {
            "text": "Disputes must be submitted within thirty days.",
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 8,
            "distance": 0.78
        }
    ]

    # Mock the real prompt, LLM call, and retrieval
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

    # Check HTTP response status
    assert response.status_code == 200

    # Read the JSON response
    data = response.json()

    # Check the question
    assert data["question"] == (
        "What is the card dispute deadline?"
    )

    # Check the generated answer
    assert data["answer"] == "The deadline is 30 days."

    # Check that a generated answer has the correct status
    assert data["answer_status"] == "generated"

    # Check that source information is included
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

    # Simulate Ollama being offline
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

    # Check that the API returns Service Unavailable
    assert response.status_code == 503

    # Check the error message
    assert response.json() == {
        "detail": "Ollama is unavailable. Check that it is running."
    }


# ==========================================
# TEST 6: Ensure retrieval happens only once
# ==========================================

def test_retrieval_happens_only_once():

    question = "What is the card dispute deadline?"

    # Simulate a retrieved banking passage
    fake_passages = [
        {
            "text": "Disputes must be submitted within thirty days.",
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 8,
            "distance": 0.78
        }
    ]

    # Simulate Qwen's answer
    fake_llm_response = BytesIO(
        b'{"response": "The deadline is 30 days."}'
    )

    # Mock retrieval, prompt building, and Ollama
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

    # Verify the request succeeded
    assert response.status_code == 200

    # Verify retrieval happened exactly once
    mock_retrieve.assert_called_once_with(
        question,
        n_results=3
    )

    # Verify the prompt reused the same passages
    mock_build_prompt.assert_called_once_with(
        question,
        passages=fake_passages
    )


# ==========================================
# TEST 7: Verify API health check
# ==========================================

def test_health_endpoint():

    # Send a GET request to the health endpoint
    response = client.get("/health")

    # Check that the endpoint responds successfully
    assert response.status_code == 200

    # Check the response content
    assert response.json() == {
        "status": "ok",
        "service": "Bank Policy RAG API"
    }


# ==========================================
# TEST 8: Verify unsupported answer status
# ==========================================

def test_unsupported_answer_status():

    # Simulate Qwen declining to answer
    fake_llm_response = BytesIO(
        b'{"response": "I cannot confirm the answer from the provided documents."}'
    )

    # Simulate a retrieved passage that does not answer the question
    fake_passages = [
        {
            "text": "General banking terms and conditions.",
            "source": "cb-retail-terms-and-conditions-en.pdf",
            "page": 6,
            "distance": 0.85
        }
    ]

    # Mock retrieval, prompt generation, and Ollama
    with (
        patch("main.build_prompt", return_value="Test prompt"),
        patch("main.urlopen", return_value=fake_llm_response),
        patch("main.retrieve_passages", return_value=fake_passages)
    ):

        response = client.post(
            "/ask",
            json={
                "question": "How many days of paid annual leave do bank employees receive?"
            }
        )

    # Verify the response
    assert response.status_code == 200

    data = response.json()

    # Verify that Qwen declined to answer
    assert data["answer"] == (
        "I cannot confirm the answer from the provided documents."
    )

    # Verify the new answer status
    assert data["answer_status"] == "not_confirmed"
