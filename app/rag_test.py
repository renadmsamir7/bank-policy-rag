
import json
from urllib.request import Request, urlopen

from prompt_builder import build_prompt


# 1. Ask a banking question
question = input("Ask a banking policy question: ").strip()

print("Question:", question)
print("\nRetrieving banking documents...")

# 2. Retrieve evidence and build our RAG prompt
prompt = build_prompt(question)

# 3. Prepare a request for our local Ollama model
data = {
    "model": "qwen2.5:3b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0
    }
}

request = Request(
    url="http://localhost:11434/api/generate",
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

# 4. Send the prompt to Qwen
print("Generating answer with Qwen...")

with urlopen(request, timeout=300) as response:
    result = json.load(response)

# 5. Display the generated answer
print("\n--- AI ANSWER ---")
print(result["response"].strip())
