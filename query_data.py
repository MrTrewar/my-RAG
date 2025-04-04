import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import pprint

# --- Konfiguration ---
chroma_db_path = "chroma_db" # Pfad zur gespeicherten ChromaDB
collection_name = "scientific_papers" # Name der Sammlung
# Dasselbe Modell wie beim Indizieren verwenden!
model_name = 'all-MiniLM-L6-v2'
# Pfad zu den JSON-Dateien mit den Chunks (Text + Metadaten)
chunks_input_directory = "chunks_output"

# Beispiel-Nutzeranfrage
user_query = "What were the results regarding CAR T cell persistence?"
# Anzahl der Top-Ergebnisse, die zurückgegeben werden sollen
top_k = 5
# --------------------

# --- Initialisierung ---
print("Initializing ChromaDB client...")
try:
    client = chromadb.PersistentClient(path=chroma_db_path)
except Exception as e:
    print(f"Error initializing ChromaDB client at path '{chroma_db_path}': {e}")
    exit()

print(f"Getting collection: {collection_name}")
try:
    collection = client.get_collection(name=collection_name)
except Exception as e:
    print(f"Error getting collection '{collection_name}'. Does it exist in '{chroma_db_path}'?")
    print(f"Error details: {e}")
    exit()

print(f"Loading sentence transformer model: {model_name}")
try:
    model = SentenceTransformer(model_name)
except Exception as e:
    print(f"Error loading model {model_name}. Do you have internet access?")
    print(f"Error details: {e}")
    exit()
print("Model loaded.")

if not os.path.exists(chunks_input_directory):
    print(f"Error: Chunks input directory '{chunks_input_directory}' not found. Cannot retrieve chunk text.")
    exit()

# --- Anfrage und Retrieval ---
print(f"\nUser Query: '{user_query}'")

print("Generating embedding for the query...")
query_embedding = model.encode(user_query).tolist()

print(f"Querying ChromaDB for top {top_k} most similar chunks...")
try:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['metadatas', 'distances'] # 'documents' holen wir selbst
    )
except Exception as e:
    print(f"Error querying ChromaDB collection: {e}")
    exit()

# --- Ergebnisse verarbeiten und Text nachladen ---
print("\nQuery Results:")

loaded_chunks_cache = {} # Cache für geladene JSON-Dateien

if results and results.get('ids') and results['ids'][0]:
    num_results_found = len(results['ids'][0])
    print(f"Found {num_results_found} results:")

    for i in range(num_results_found):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        doc_id = results['ids'][0][i]
        document_text = "Error: Text not found." # Standardwert

        source_file = metadata.get('source_file')
        if source_file:
            # Rekonstruiere den Namen der JSON-Datei
            json_filename = source_file.replace(".pdf", "_chunks.json")
            chunks_json_path = os.path.join(chunks_input_directory, json_filename)

            try:
                # Lade Chunk-Daten aus Cache oder Datei
                if chunks_json_path not in loaded_chunks_cache:
                    print(f"    (Loading chunks from {json_filename}...)")
                    if os.path.exists(chunks_json_path):
                        with open(chunks_json_path, 'r', encoding='utf-8') as f_json:
                            loaded_chunks_cache[chunks_json_path] = json.load(f_json)
                    else:
                         loaded_chunks_cache[chunks_json_path] = None # Markiere als nicht gefunden

                # Hole Daten aus Cache
                chunks_list = loaded_chunks_cache.get(chunks_json_path)

                if chunks_list:
                    # Finde den passenden Chunk anhand der Metadaten
                    found_chunk = None
                    for chunk in chunks_list:
                        # Vergleiche die Metadaten-Dictionaries
                        if chunk.get('metadata') == metadata:
                            found_chunk = chunk
                            break

                    if found_chunk:
                        document_text = found_chunk.get('text', "Error: Text key missing in chunk.")
                    else:
                        document_text = "Error: Matching chunk not found in JSON."
                else:
                     document_text = f"Error: Chunks file '{json_filename}' not found or empty."

            except json.JSONDecodeError:
                document_text = f"Error: Could not decode JSON from '{json_filename}'."
            except Exception as e:
                 document_text = f"Error loading/processing chunks file: {e}"
        else:
            document_text = "Error: 'source_file' missing in metadata."


        # --- Ausgabe ---
        print("-" * 20)
        print(f"Rank {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Metadata:")
        pprint.pprint(metadata, indent=4)
        print(f"  Retrieved Text: {document_text[:500]}...") # Zeige Anfang des Textes

else:
    print("No results found for the query.")

print("\nRetrieval finished.")