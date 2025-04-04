import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np # Wird oft für die Arbeit mit Vektoren verwendet

# --- Konfiguration ---
chunks_input_directory = "chunks_output"  # Ordner mit den Chunk-JSON-Dateien
embeddings_output_directory = "embeddings_output" # Neuer Ordner für Daten mit Embeddings
# Wähle ein Sentence Transformer Modell:
# 'all-MiniLM-L6-v2': Schnell, gut für allgemeine Ähnlichkeit.
# 'multi-qa-mpnet-base-dot-v1': Gut für semantische Suche / Q&A.
# Für wissenschaftliche Texte könnten spezifischere Modelle besser sein (ggf. später testen)
model_name = 'all-MiniLM-L6-v2'
# --------------------

print(f"Loading Sentence Transformer model: {model_name}")
# Lädt das Modell. Beim ersten Mal wird es heruntergeladen (kann dauern).
# Stelle sicher, dass du Internetzugang hast, wenn du das Skript zum ersten Mal ausführst.
try:
    model = SentenceTransformer(model_name)
except Exception as e:
    print(f"Error loading model {model_name}. Do you have internet access?")
    print(f"Error details: {e}")
    exit()
print("Model loaded.")

# Erstelle das Ausgabe-Verzeichnis für Embeddings, falls es nicht existiert
if not os.path.exists(embeddings_output_directory):
    os.makedirs(embeddings_output_directory)
    print(f"Created embeddings output directory: {embeddings_output_directory}")

if not os.path.exists(chunks_input_directory):
    print(f"Error: Chunks input directory '{chunks_input_directory}' not found.")
    exit()

print(f"Starting embedding generation from '{chunks_input_directory}'...")

# Gehe durch alle JSON-Dateien im Chunk-Verzeichnis
for filename in os.listdir(chunks_input_directory):
    if filename.lower().endswith("_chunks.json"):
        json_path = os.path.join(chunks_input_directory, filename)
        # Erstelle einen Basisnamen für die Output-Datei
        base_output_name = filename.replace("_chunks.json", "")
        # Wir speichern die Embeddings separat als NumPy-Datei und die Metadaten weiterhin als JSON
        output_meta_path = os.path.join(embeddings_output_directory, f"{base_output_name}_meta.json")
        output_embed_path = os.path.join(embeddings_output_directory, f"{base_output_name}_embeddings.npy")

        print(f"  Processing: {filename} ...")

        try:
            # Lade die Chunk-Daten aus der JSON-Datei
            with open(json_path, 'r', encoding='utf-8') as f_in:
                chunks_data = json.load(f_in)

            if not chunks_data:
                print("    -> Warning: No chunks found in this file. Skipping.")
                continue

            # Extrahiere nur die Texte für das Embedding-Modell
            texts_to_embed = [chunk['text'] for chunk in chunks_data]

            # Generiere die Embeddings für alle Texte in dieser Datei auf einmal (effizienter)
            print(f"    Generating embeddings for {len(texts_to_embed)} chunks...")
            embeddings = model.encode(texts_to_embed, show_progress_bar=True) # Zeigt Fortschrittsbalken
            print(f"    Embeddings generated with shape: {embeddings.shape}") # Zeigt Dimension (Anzahl Chunks, Vektorlänge)

            # --- Speichern der Ergebnisse ---
            # 1. Speichere die Embeddings als NumPy-Datei (.npy)
            np.save(output_embed_path, embeddings)

            # 2. Speichere die Metadaten (und Texte, falls gewünscht) als JSON
            #    Wir erstellen hier eine Liste, die nur die Metadaten enthält,
            #    korrespondierend zu den Zeilen in der .npy-Datei.
            metadata_list = [chunk['metadata'] for chunk in chunks_data]
            # Optional: Füge Text hinzu, wenn du ihn hier brauchst
            # metadata_list = [{'text': chunk['text'], 'metadata': chunk['metadata']} for chunk in chunks_data]

            with open(output_meta_path, 'w', encoding='utf-8') as f_meta_out:
                json.dump(metadata_list, f_meta_out, ensure_ascii=False, indent=2)

            print(f"    -> Success! Saved embeddings to '{output_embed_path}' and metadata to '{output_meta_path}'")

        except json.JSONDecodeError as e:
            print(f"    -> Error: Could not decode JSON from {filename}. Error: {e}")
        except Exception as e:
             print(f"    -> An unexpected error occurred processing {filename}: {e}")


print("\nEmbedding generation finished.")
print(f"Embedding data is in the '{embeddings_output_directory}' folder.")