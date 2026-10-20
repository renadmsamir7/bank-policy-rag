
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# 1. Connect to our existing vector database
client = chromadb.PersistentClient(path="chroma_db")

# 2. Open our banking policy collection
collection = client.get_collection(
    name="bank_policies",
    embedding_function=DefaultEmbeddingFunction()
)

# 3. Ask our first banking question
question = "What is the deadline for disputing a card transaction?"

print("Question:", question)

# 4. Search for the 3 most relevant text chunks
results = collection.query(
    query_texts=[question],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

# 5. Display the search results
for i, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),
    start=1
):
    print(f"\n--- Result {i} ---")
    print("Source:", metadata["file_name"])
    print("Page:", metadata["page"])
    print("Distance:", round(distance, 4))
    print("Text:")
    print(document)
