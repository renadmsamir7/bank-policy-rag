
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


API_URL = "http://127.0.0.1:8001/ask"

EVAL_FILE = Path(__file__).with_name("answer_eval_cases.json")

with open(EVAL_FILE, "r", encoding="utf-8") as file:
    test_cases = json.load(file)


# ==========================================
# STEP 1: Check expected answer terms
# ==========================================

def term_matches(term, answer):
    """Check an expected term, allowing a few equivalent wordings."""

    term = term.lower()
    answer = answer.lower()

    if term == "thirty days":
        return "thirty days" in answer or "30 days" in answer

    if term == "replacing":
        return "replacing" in answer or "replacement" in answer

    return term in answer


# ==========================================
# STEP 2: Send a question to our API
# ==========================================

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


# ==========================================
# STEP 3: Verify source citations
# ==========================================

def check_source_citation(answer, sources, expected_source, expected_page):
    """
    Verify that the answer cites the expected retrieved source.

    Accept either:
    1. A matching source number, such as [Source 2].
    2. The expected PDF filename together with its page number.
    """

    normalized_answer = answer.casefold()

    for source in sources:

        # Identify the expected retrieved source.
        if (
            source.get("file_name") != expected_source
            or source.get("page") != expected_page
        ):
            continue

        # Check for a source number such as "Source 2".
        rank = source.get("rank")

        source_number_cited = (
            rank is not None
            and re.search(
                rf"\bsource\s*{re.escape(str(rank))}\b",
                normalized_answer
            ) is not None
        )

        # Check for the actual filename and page number.
        filename_cited = (
            expected_source.casefold() in normalized_answer
        )

        page_cited = re.search(
            rf"\bpage\s*:?\s*{re.escape(str(expected_page))}\b",
            normalized_answer
        ) is not None

        if source_number_cited or (filename_cited and page_cited):
            return True

    return False


# ==========================================
# STEP 4: Check answer formatting
# ==========================================

def check_answer_formatting(answer):
    """
    Detect known merged-word errors in generated answers.

    Examples:
    - whichthe
    - withinthirty
    - Thisfee

    This is a targeted formatting check,
    not a complete grammar checker.
    """

    merged_words = (
        "whichthe",
        "withinthirty",
        "thisfee"
    )

    normalized_answer = answer.casefold()

    formatting_errors = [
        word
        for word in merged_words
        if word in normalized_answer
    ]

    return len(formatting_errors) == 0


# ==========================================
# STEP 5: Evaluate generated answers
# ==========================================

def evaluate_answers():
    """
    Evaluate:
    1. Answer correctness
    2. Source retrieval
    3. Source citations
    4. Answer formatting
    5. Answer status
    """

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

        # ==========================================
        # Extract the API response
        # ==========================================

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

            # Check that the answer cites the expected source.
            correct_citation = check_source_citation(
                answer=answer,
                sources=sources,
                expected_source=case["expected_source"],
                expected_page=case["expected_page"]
            )

            # NEW: Check for known merged-word formatting errors.
            correct_formatting = check_answer_formatting(answer)

            # Supported answers should have generated status.
            expected_status = "generated"

            # Every check must pass.
            success = (
                correct_content
                and correct_source
                and correct_citation
                and correct_formatting
            )

            print("Correct answer content:", correct_content)
            print("Expected source retrieved:", correct_source)
            print("Expected source cited in answer:", correct_citation)
            print("Correct answer formatting:", correct_formatting)

        # ==========================================
        # Evaluate unsupported questions
        # ==========================================

        elif case["answer_type"] == "unsupported":

            # Check that the model declined to answer.
            expected_response = (
                case["expected_response"].strip().lower()
            )

            correct_refusal = (
                expected_response in answer.strip().lower()
            )

            # Unsupported answers should have not_confirmed status.
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

        # The test passes only if all relevant checks pass.
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
    # STEP 6: Calculate overall pass rate
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
