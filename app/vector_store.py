
import chromadb

# 1. Create a local ChromaDB database
client = chromadb.PersistentClient(
    path="chroma_db"
)

# 2. Create a collection for banking policies
collection = client.get_or_create_collection(
    name="bank_policies"
)

# 3. Check our collection
print("Collection name:", collection.name)
print("Stored chunks:", collection.count())
