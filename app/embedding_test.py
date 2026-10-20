
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Load the embedding model
embedding_model = DefaultEmbeddingFunction()

# Our first banking-related sentence
sentence = "What are the fees for opening a bank account?"

# Convert the sentence into an embedding
embeddings = embedding_model([sentence])

# Display the result
print("Number of embeddings:", len(embeddings))
print("Dimensions:", len(embeddings[0]))
print("First 5 numbers:", embeddings[0][:5])
