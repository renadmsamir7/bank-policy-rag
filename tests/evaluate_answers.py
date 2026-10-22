
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
    """Evaluate answer content and retrieved source metadata."""

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

        answer = result.get("answer", "")
        sources = result.get("retrieved_sources", [])

        print("ANSWER:", answer)

        if case["answer_type"] == "supported":

            # Verify expected information appears in the answer.
            correct_content = all(
                term_matches(term, answer)
                for term in case["expected_terms"]
            )

            # Verify the expected document and page were retrieved.
            correct_source = any(
                source.get("file_name") == case["expected_source"]
                and source.get("page") == case["expected_page"]
                for source in sources
            )

            success = correct_content and correct_source

            print("Correct answer content:", correct_content)
            print("Expected source retrieved:", correct_source)

        elif case["answer_type"] == "unsupported":

            expected = case["expected_response"].strip().lower()
            success = expected in answer.strip().lower()

            print("Correct refusal:", success)

        else:
            success = False
            print("Unknown evaluation case type.")

        if success:
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

    pass_rate = passed / total if total else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Answer evaluation pass rate: {pass_rate:.2%}")

    return pass_rate


if __name__ == "__main__":
    evaluate_answers()
