
import json
import chromadb

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# 1. Load our saved banking chunks
with open("data/processed_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks)}")

# 2. Connect to our existing ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

embedding_model = DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="bank_policies",
    embedding_function=embedding_model
)

# 3. Choose how many chunks to process at once
batch_size = 50

# 4. Process our chunks in batches
for start in range(0, len(chunks), batch_size):

    batch = chunks[start:start + batch_size]

    # Unique ID for every chunk
    ids = [
        f"chunk_{i}"
        for i in range(start, start + len(batch))
    ]

    # Original text of each chunk
    documents = [
        chunk["text"]
        for chunk in batch
    ]

    # PDF source information
    metadatas = [
        {
            "file_name": chunk["file_name"],
            "page": chunk["page"]
        }
        for chunk in batch
    ]

    # Generate embeddings and store in ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Stored {start + len(batch)} / {len(chunks)} chunks")

# 5. Verify the final result
print(f"\nTotal chunks in ChromaDB: {collection.count()}")
print("All banking chunks stored successfully!")
