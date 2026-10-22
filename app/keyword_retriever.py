
import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


QUESTION_WORDS = {
    "according", "to", "the", "of", "what", "is",
    "are", "for", "a", "an", "how", "much",
    "does", "do", "can", "you", "please",
    "me", "tell", "about", "my", "in"
}

GENERAL_BANK_WORDS = {
    "bank", "tariff", "charges", "fee"
}


def focus_query(question):
    """Extract important search terms from a natural-language question."""

    words = tokenize(question)

    useful_words = [
        word for word in words
        if word not in QUESTION_WORDS
    ]

    specific_words = [
        word for word in useful_words
        if word not in GENERAL_BANK_WORDS
    ]

    return specific_words if len(specific_words) >= 2 else useful_words


def keyword_search(question, collection, n_results=5):
    """Retrieve banking passages using BM25 keyword matching."""

    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document) for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    # Focus on important words instead of the entire question.
    search_terms = focus_query(question)
    scores = bm25.get_scores(search_terms)

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: scores[i],
        reverse=True
    )

    passages = []

    for i in ranked_indices[:n_results]:
        passages.append({
            "text": documents[i],
            "source": metadatas[i]["file_name"],
            "page": metadatas[i]["page"],
            "bm25_score": float(scores[i])
        })

    return passages
