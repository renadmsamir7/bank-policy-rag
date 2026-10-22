
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


API_URL = "http://127.0.0.1:8001/ask"

EVAL_FILE = Path(__file__).with_name("answer_eval_cases.json")

with open(EVAL_FILE, "r", encoding="utf-8") as file:
    test_cases = json.load(file)


def term_matches(term, answer):
    """Check an expected term, allowing a few equivalent wordings."""

    term = term.lower()
    answer = answer.lower()

    if term == "thirty days":
        return "thirty days" in answer or "30 days" in answer

    if term == "replacing":
        return "replacing" in answer or "replacement" in answer

    return term in answer


def ask_api(question):
    """Send a question to the FastAPI RAG application."""

    payload = json.dumps({
        "question": question
    }).encode("utf-8")

    request = Request(
        url=API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(request, timeout=300) as response:
        return json.load(response)


def evaluate_answers():
    """Evaluate answer content, retrieved sources, and answer status."""

    total = len(test_cases)
    passed = 0

    print("\n=== Bank Policy RAG Answer Evaluation ===")

    for case in test_cases:
        print(f"\nQUESTION: {case['question']}")

        try:
            result = ask_api(case["question"])

        except (URLError, HTTPError, TimeoutError, ValueError) as error:
            print("RESULT: FAIL")
            print("API error:", error)
            continue

        # Extract information from the API response.
        answer = result.get("answer", "")
        sources = result.get("retrieved_sources", [])
        status = result.get("answer_status")

        print("ANSWER:", answer)
        print("ANSWER STATUS:", status)

        # ==========================================
        # Evaluate supported questions
        # ==========================================

        if case["answer_type"] == "supported":

            # Check that the answer includes expected information.
            correct_content = all(
                term_matches(term, answer)
                for term in case["expected_terms"]
            )

            # Check that the expected document and page were retrieved.
            correct_source = any(
                source.get("file_name") == case["expected_source"]
                and source.get("page") == case["expected_page"]
                for source in sources
            )

            # Supported answers should have a generated status.
            expected_status = "generated"

            success = correct_content and correct_source

            print("Correct answer content:", correct_content)
            print("Expected source retrieved:", correct_source)

        # ==========================================
        # Evaluate unsupported questions
        # ==========================================

        elif case["answer_type"] == "unsupported":

            # Check that the model declined to provide an unsupported answer.
            expected_response = (
                case["expected_response"].strip().lower()
            )

            correct_refusal = (
                expected_response in answer.strip().lower()
            )

            # Unsupported answers should have a not_confirmed status.
            expected_status = "not_confirmed"

            success = correct_refusal

            print("Correct refusal:", correct_refusal)

        # ==========================================
        # Handle unknown evaluation types
        # ==========================================

        else:
            expected_status = None
            success = False

            print("Unknown evaluation case type.")

        # ==========================================
        # Verify answer status
        # ==========================================

        status_correct = (
            expected_status is not None
            and status == expected_status
        )

        print("Expected answer status:", expected_status)
        print("Correct answer status:", status_correct)

        # The test passes only if the original checks
        # and the answer status check both pass.
        success = success and status_correct

        # ==========================================
        # Record the test result
        # ==========================================

        if success:
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

    # ==========================================
    # Calculate overall pass rate
    # ==========================================

    pass_rate = passed / total if total else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Answer evaluation pass rate: {pass_rate:.2%}")

    return pass_rate


if __name__ == "__main__":
    evaluate_answers()
