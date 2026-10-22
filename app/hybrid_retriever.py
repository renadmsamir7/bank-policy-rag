
from retriever import retrieve_passages, collection
from keyword_retriever import keyword_search


def hybrid_search(question, n_results=3, candidates=10):
    """Combine semantic and BM25 retrieval using reciprocal rank fusion."""

    semantic_results = retrieve_passages(
        question,
        n_results=candidates
    )

    keyword_results = keyword_search(
        question,
        collection,
        n_results=candidates
    )

    combined = {}
    rrf_k = 60

    for results in [semantic_results, keyword_results]:
        weight = 2.0 if results is keyword_results else 1.0
        for rank, passage in enumerate(results, start=1):

            # Identify each unique document chunk.
            key = (
                passage["source"],
                passage["page"],
                passage["text"]
            )

            if key not in combined:
                combined[key] = {
                    "text": passage["text"],
                    "source": passage["source"],
                    "page": passage["page"],
                    "rrf_score": 0.0
                }

            # Higher-ranked passages receive larger contributions.
            combined[key]["rrf_score"] += weight / (rrf_k + rank)

    ranked_passages = sorted(
        combined.values(),
        key=lambda passage: passage["rrf_score"],
        reverse=True
    )


    # Preserve the strongest result from each retrieval method.
    selected = []
    seen = set()

    priority_passages = (
        semantic_results[:1]
        + keyword_results[:1]
        + ranked_passages
    )

    for passage in priority_passages:
        key = (
            passage["source"],
            passage["page"],
            passage["text"]
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(combined[key])

        if len(selected) >= n_results:
            break

    return selected
