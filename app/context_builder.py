
import json
from functools import lru_cache
from pathlib import Path

from retriever import retrieve_passages


# Location of our processed banking policy chunks.
CHUNKS_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed_chunks.json"
)


@lru_cache(maxsize=1)
def load_chunks():
    """Load processed chunks once for neighboring-context lookup."""

    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        # Keep the existing RAG behavior if the file is unavailable.
        return []


def expand_passage(passage, chunks):
    """
    Include the previous chunk when a retrieved passage
    appears to begin in the middle of a sentence.
    """

    text = passage["text"]
    source = passage["source"]
    page = passage["page"]

    for index, chunk in enumerate(chunks):

        # Find the original retrieved chunk.
        if (
            chunk.get("file_name") != source
            or str(chunk.get("page")) != str(page)
            or " ".join(chunk.get("text", "").split())
            != " ".join(text.split())
        ):
            continue

        # No previous chunk exists.
        if index == 0:
            break

        previous = chunks[index - 1]

        # Only combine chunks from the same PDF page.
        if (
            previous.get("file_name") != source
            or str(previous.get("page")) != str(page)
        ):
            break

        previous_text = previous.get("text", "").strip()
        current_text = text.strip()

        # A lowercase start and an unfinished preceding
        # sentence suggest the text was split across chunks.
        if (
            previous_text
            and current_text
            and current_text[0].islower()
            and previous_text[-1] not in ".!?:"
        ):
            return previous_text + "\n" + current_text

        break

    return text


def build_context(question, n_results=3, passages=None):
    """Build a source-labeled context for the language model."""

    # Retrieve passages only when none were provided.
    if passages is None:
        passages = retrieve_passages(
            question,
            n_results=n_results
        )

    chunks = load_chunks()

    context_parts = []

    # Organize each passage with its source.
    for i, passage in enumerate(passages, start=1):

        # Expand passages that begin mid-sentence.
        expanded_text = expand_passage(passage, chunks)

        section = (
            f"[Source {i}]\n"
            f"File: {passage['source']}\n"
            f"Page: {passage['page']}\n"
            f"Text:\n{expanded_text}"
        )

        context_parts.append(section)

    # Combine passages into one context.
    context = "\n\n".join(context_parts)

    return context


# Test our context builder.
if __name__ == "__main__":

    question = "What is the deadline for disputing a card transaction?"

    context = build_context(question)

    print("QUESTION:")
    print(question)

    print("\nRETRIEVED CONTEXT:")
    print(context)
