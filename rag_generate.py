import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import ollama # Ollama Bibliothek importieren
import pprint
import time
import traceback

# --- Konfiguration ---
chroma_db_path = "chroma_db"
collection_name = "scientific_papers"
embedding_model_name = 'all-MiniLM-L6-v2' # Für die Query
chunks_input_directory = "chunks_output"
# Ollama Konfiguration
# WICHTIG: Verwende qwen:0.5b für 8GB RAM
ollama_model_name = "llama3:8b"
ollama_base_url = "http://localhost:11434"
# Anzahl der Chunks für den Kontext (weniger ist oft besser für kleine Modelle)
retrieval_k = 3
# --------------------

# --- Initialisierung ---
print("Initializing components...")
try: # <<< OUTERMOST TRY BLOCK: For all initializations >>>
    # --- Component Initialization ---
    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    collection = chroma_client.get_collection(name=collection_name)
    print("ChromaDB initialized.")

    print("Initializing Sentence Transformer...")
    embedding_model = SentenceTransformer(embedding_model_name)
    print("Sentence Transformer initialized.")

    print("Initializing Ollama Client...")
    ollama_client = ollama.Client(host=ollama_base_url)
    print("Ollama Client initialized.")

    # --- Ollama Connection and Model Check ---
    print(f"Checking Ollama connection and model '{ollama_model_name}'...")
    try: # <<< MIDDLE TRY BLOCK: For Ollama API interaction >>>
        # --- STEP 1: Call Ollama API to get the list ---
        response = ollama_client.list()
        # print(f"DEBUG: Ollama response type: {type(response)}") # Optional Debug
        # print(f"DEBUG: Ollama response value: {response}")     # Optional Debug

        # --- STEP 2: Try to parse the response received ---
        try: # <<< INNERMOST TRY BLOCK: For parsing the response >>>
             available_models = [m.model for m in response.models]

             # Check if the desired model or a variant exists
             model_found = any(ollama_model_name in model_tag for model_tag in available_models)
             if not model_found:
                  print(f"Error: Model like '{ollama_model_name}' not found in Ollama.")
                  print(f"Available models: {available_models}")
                  print(f"Please pull the model using: ollama run {ollama_model_name}")
                  exit(1) # Exit with error code
             print(f"Ollama connection successful and model '{ollama_model_name}' found.")
             print(f"Full list of available models: {available_models}")

        except AttributeError:
             print("Error: Unexpected response format from Ollama. Could not find 'models' attribute.")
             print(f"Received response structure: {type(response)}")
             pprint.pprint(response)
             traceback.print_exc()
             exit(1) # Exit with error code
        except Exception as e:
             print(f"Error parsing Ollama models list: {e}")
             traceback.print_exc()
             exit(1) # Exit with error code

    except Exception as e: # <<< MIDDLE EXCEPT BLOCK: Handles errors during ollama_client.list() >>>
         print(f"Error communicating with Ollama during model check: {e}")
         traceback.print_exc()
         exit(1) # Exit with error code

    # --- If we reach here, all checks inside the Ollama block passed ---
    print("Ollama check completed successfully.")

# <<< OUTERMOST EXCEPT BLOCK: Handles errors in ANY initialization step >>>
except Exception as e:
    print(f"\n--- FATAL ERROR DURING INITIALIZATION ---")
    print(f"Error details: {e}")
    traceback.print_exc()
    print("\nPotential Causes:")
    print("- ChromaDB connection failed (check path, permissions, or if DB is running/valid).")
    print(f"- SentenceTransformer model '{embedding_model_name}' could not be downloaded/loaded.")
    print(f"- Ollama client could not connect to '{ollama_base_url}' (is Ollama running?).")
    print("- An error occurred during the Ollama model check process (see details above).")
    print("-----------------------------------------")
    exit(1) # Exit with error code

# --- If we reach here, the ENTIRE outer try block succeeded ---
print("\nComponents initialized successfully.")

# --- Cache for loaded Chunk-Texte (keep this outside the try block) ---
loaded_chunks_cache = {}


# Cache für geladene Chunk-Texte
loaded_chunks_cache = {}

def get_chunk_text(metadata):
    """ Lädt den Text für einen Chunk anhand seiner Metadaten nach. """
    global loaded_chunks_cache
    document_text = "Error: Could not retrieve text."
    source_file = metadata.get('source_file')
    if source_file:
        json_filename = source_file.replace(".pdf", "_chunks.json")
        chunks_json_path = os.path.join(chunks_input_directory, json_filename)
        try:
            if chunks_json_path not in loaded_chunks_cache:
                if os.path.exists(chunks_json_path):
                    with open(chunks_json_path, 'r', encoding='utf-8') as f_json:
                        loaded_chunks_cache[chunks_json_path] = json.load(f_json)
                else:
                     loaded_chunks_cache[chunks_json_path] = None
            chunks_list = loaded_chunks_cache.get(chunks_json_path)
            if chunks_list:
                p_index = metadata.get('paragraph_index')
                c_type = metadata.get('chunk_type')
                found_chunk = None
                for chunk in chunks_list:
                    chunk_meta = chunk.get('metadata', {})
                    if chunk_meta.get('paragraph_index') == p_index and chunk_meta.get('chunk_type') == c_type:
                        found_chunk = chunk
                        break
                if not found_chunk:
                    found_chunk = next((chunk for chunk in chunks_list if chunk.get('metadata') == metadata), None)
                if found_chunk:
                    document_text = found_chunk.get('text', "Error: Text key missing.")
                else:
                    document_text = f"Error: Matching chunk not found (Index: {p_index}, Type: {c_type})."
            else:
                 document_text = f"Error: Chunks file '{json_filename}' not found or empty."
        except Exception as e:
             document_text = f"Error loading/processing chunks file '{json_filename}': {e}"
    else:
        document_text = "Error: 'source_file' missing in metadata."
    return document_text

# --- Haupt-Schleife für Anfragen ---
while True:
    user_query = input("\nStelle eine Frage (oder 'exit' zum Beenden): ")
    if user_query.lower() == 'exit':
        break
    if not user_query:
        continue

    # 1. Retrieval
    print(f"\n1. Retrieving top {retrieval_k} relevant chunks...")
    try:
        query_embedding = embedding_model.encode(user_query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_k, # Verwende konfigurierte Anzahl
            include=['metadatas', 'distances']
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        continue

    if not results or not results.get('ids') or not results['ids'][0]:
        print("Could not find relevant chunks for this query.")
        continue

    # 2. Kontext aufbereiten
    print("2. Preparing context from retrieved chunks...")
    context_parts = []
    retrieved_sources = set()
    print("   Retrieved Chunks (Metadata & Text Snippet):")
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        chunk_text = get_chunk_text(metadata)
        source = metadata.get('source_file', 'Unknown Source')
        retrieved_sources.add(source)
        print(f"   - Rank {i+1} (Dist: {distance:.4f}): {source} (Index: {metadata.get('paragraph_index', 'N/A')}, Type: {metadata.get('chunk_type', 'N/A')})")
        if not chunk_text.startswith("Error:"):
             print(f"     Text: {chunk_text[:150]}...")
        else:
             print(f"     {chunk_text}")
        context_parts.append(f"Source: {source}, Paragraph {metadata.get('paragraph_index', 'N/A')}:\n{chunk_text}\n")
    context_string = "\n---\n".join(context_parts)

    # 3. Prompt erstellen (Angepasst für kleinere Modelle, etwas einfacher)
    prompt = f"""Kontext:
---
{context_string}
---
Frage: {user_query}

Anweisung: Beantworte die Frage kurz und nur mit Informationen aus dem obigen Kontext. Wenn die Antwort nicht im Kontext steht, sage "Die Antwort ist nicht im Kontext enthalten.".

Antwort:"""

    # 4. Generation mit Ollama
    print("\n3. Generating answer with Ollama (Model: {})...".format(ollama_model_name))
    print("   (Dies kann mit einem kleinen Modell auf 8GB RAM eine Weile dauern...)")
    start_time = time.time()
    try:
        response = ollama_client.chat(
            model=ollama_model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
             options={
                 'temperature': 0.2, # Noch deterministischer für kleine Modelle
                 'num_predict': 150 # Kürzere Antwort erwarten
             }
        )
        generated_answer = response['message']['content']
        end_time = time.time()
        print(f"   (Generation took {end_time - start_time:.2f} seconds)")

    except Exception as e:
        print(f"\nError calling Ollama API: {e}")
        generated_answer = "Fehler bei der Antwortgenerierung mit Ollama."
        end_time = time.time()
        print(f"   (Attempt took {end_time - start_time:.2f} seconds)")


    print("\n--- Generierte Antwort ---")
    print(generated_answer)
    print("-------------------------")
    print(f"(Verwendete Quellen im Kontext: {', '.join(sorted(list(retrieved_sources)))})")


print("\nProgramm beendet.")