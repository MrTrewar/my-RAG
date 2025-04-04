import ollama
import pprint
import os

# Ensure this matches the default or your server's actual address
# It will also check the OLLAMA_HOST environment variable
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
print(f"Attempting to connect to Ollama at: {ollama_host}")

try:
    # Explicitly use the determined host
    client = ollama.Client(host=ollama_host)
    print("Client created. Attempting to list models...")
    response = client.list()
    print("\n--- Raw Response from ollama.list() ---")
    pprint.pprint(response)
    print("---------------------------------------")

    if 'models' in response and isinstance(response['models'], list):
         available_models = [m.model for m in response.models] # Access attributes directly
         print(f"\nParsed available models: {available_models}")
    else:
         print("\nCould not parse models from the response.")

except Exception as e:
    print(f"\nError connecting to Ollama or listing models: {e}")

print("\nTest script finished.")