
import json
from pathlib import Path

from hybrid_retriever import hybrid_search


# Load the evaluation questions
EVAL_FILE = Path(__file__).with_name("rag_eval_cases.json")

with open(EVAL_FILE, "r", encoding="utf-8") as file:
    test_cases = json.load(file)


def evaluate_retrieval():
    """Measure retrieval Hit Rate@3."""

    total_tests = len(test_cases)
    passed_tests = 0

    print("\n=== Bank Policy RAG Evaluation ===")

    for case in test_cases:
        question = case["question"]

        # Retrieve the three best passages
        passages = hybrid_search(question, n_results=3)

        correct_rank = None

        for rank, passage in enumerate(passages, start=1):
            correct_source = (
                passage["source"] == case["expected_source"]
            )

            correct_page = (
                passage["page"] == case["expected_page"]
            )

            text = passage["text"].lower()

            correct_content = all(
                phrase.lower() in text
                for phrase in case["expected_phrases"]
            )

            if correct_source and correct_page and correct_content:
                correct_rank = rank
                break

        if correct_rank is not None:
            passed_tests += 1
            print(f"\nPASS: {case['id']}")
            print(f"Correct evidence found at rank {correct_rank}")
        else:
            print(f"\nFAIL: {case['id']}")
            print("Correct evidence not found in top 3")

    # Calculate Hit Rate@3
    hit_rate = (
        passed_tests / total_tests
        if total_tests > 0
        else 0
    )

    print("\n=== Evaluation Summary ===")
    print(f"Total questions: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Hit Rate@3: {hit_rate:.2%}")

    return hit_rate


if __name__ == "__main__":
    evaluate_retrieval()
