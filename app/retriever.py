
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# Connect to our existing vector database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="bank_policies",
    embedding_function=DefaultEmbeddingFunction()
)


# Retrieve relevant passages for any question
def retrieve_passages(question, n_results=3):

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    passages = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        passages.append({
            "text": text,
            "source": metadata["file_name"],
            "page": metadata["page"],
            "distance": distance
        })

    return passages


# Test our retriever
if __name__ == "__main__":

    question = "What is the deadline for disputing a card transaction?"

    passages = retrieve_passages(question)

    print("Question:", question)

    for i, passage in enumerate(passages, start=1):

        print(f"\n--- Passage {i} ---")
        print("Source:", passage["source"])
        print("Page:", passage["page"])
        print("Text:", passage["text"])
