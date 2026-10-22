
from retriever import retrieve_passages


def build_context(question, n_results=3, passages=None):

    # Retrieve passages only if none were provided
    if passages is None:
        passages = retrieve_passages(
            question,
            n_results=n_results
        )

    context_parts = []

    # Organize each passage with its source
    for i, passage in enumerate(passages, start=1):

        section = (
            f"[Source {i}]\n"
            f"File: {passage['source']}\n"
            f"Page: {passage['page']}\n"
            f"Text:\n{passage['text']}"
        )

        context_parts.append(section)

    # Combine passages into one context
    context = "\n\n".join(context_parts)

    return context


# Test our context builder
if __name__ == "__main__":

    question = "What is the deadline for disputing a card transaction?"

    context = build_context(question)

    print("QUESTION:")
    print(question)

    print("\nRETRIEVED CONTEXT:")
    print(context)
