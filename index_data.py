import os
import json
import numpy as np
import chromadb # ChromaDB importieren
import uuid # Um eindeutige IDs für Chunks zu generieren

# --- Konfiguration ---
embeddings_input_directory = "embeddings_output" # Ordner mit Embeddings und Metadaten
chroma_db_path = "chroma_db" # Pfad, wo ChromaDB seine Daten speichern soll
collection_name = "scientific_papers" # Name für die Sammlung in ChromaDB
# --------------------

print("Initializing ChromaDB...")
# Initialisiert ChromaDB. 'PersistentClient' speichert die Daten im angegebenen Pfad.
# Wenn der Pfad nicht existiert, wird er erstellt.
client = chromadb.PersistentClient(path=chroma_db_path)

# Erstelle eine neue Sammlung oder hole eine bestehende.
# metadata={"hnsw:space": "cosine"} legt fest, dass die Kosinus-Ähnlichkeit
# für die Vektorsuche verwendet wird, was für Sentence Transformer Embeddings üblich ist.
print(f"Getting or creating collection: {collection_name}")
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"} # Wichtig für Ähnlichkeitsmaß
    )

print(f"Starting indexing process from '{embeddings_input_directory}'...")

if not os.path.exists(embeddings_input_directory):
    print(f"Error: Embeddings input directory '{embeddings_input_directory}' not found.")
    exit()

# Gehe durch alle Metadaten-Dateien im Embedding-Verzeichnis
processed_files = 0
total_chunks_added = 0
for filename in os.listdir(embeddings_input_directory):
    if filename.lower().endswith("_meta.json"):
        base_name = filename.replace("_meta.json", "")
        meta_path = os.path.join(embeddings_input_directory, filename)
        embed_path = os.path.join(embeddings_input_directory, f"{base_name}_embeddings.npy")

        print(f"  Processing file pair: {filename} and {base_name}_embeddings.npy ...")

        # Stelle sicher, dass die zugehörige .npy-Datei existiert
        if not os.path.exists(embed_path):
            print(f"    -> Warning: Corresponding embedding file '{base_name}_embeddings.npy' not found. Skipping.")
            continue

        try:
            # Lade Metadaten
            with open(meta_path, 'r', encoding='utf-8') as f_meta:
                metadata_list = json.load(f_meta)

            # Lade Embeddings
            embeddings = np.load(embed_path)

            # Überprüfe Konsistenz: Anzahl Metadaten muss Anzahl Embeddings entsprechen
            if len(metadata_list) != embeddings.shape[0]:
                print(f"    -> Error: Mismatch between number of metadata entries ({len(metadata_list)}) "
                      f"and embeddings ({embeddings.shape[0]}) in file pair for '{base_name}'. Skipping.")
                continue

            if not metadata_list:
                print("    -> Warning: No metadata/embeddings found in this file pair. Skipping.")
                continue

            # --- Daten für ChromaDB vorbereiten ---
            ids_to_add = []
            embeddings_to_add = []
            metadata_to_add = []

            for i, meta in enumerate(metadata_list):
                # Erzeuge eine eindeutige ID für jeden Chunk
                # z.B. aus Dateiname und Index
                chunk_id = f"{meta.get('source_file', base_name + '.pdf')}_chunk_{i}"
                # Alternativ: eine zufällige UUID
                # chunk_id = str(uuid.uuid4())

                ids_to_add.append(chunk_id)
                # ChromaDB erwartet Embeddings als Liste von Listen (oder NumPy-Array)
                embeddings_to_add.append(embeddings[i].tolist()) # Konvertiere NumPy-Zeile in Liste
                # Füge die Metadaten hinzu, die wir gespeichert hatten
                metadata_to_add.append(meta) # Das Metadaten-Dictionary direkt verwenden

            # --- Daten zur ChromaDB-Sammlung hinzufügen ---
            # Füge die Chunks in Batches hinzu (hier alle auf einmal, für große Datenmengen evtl. aufteilen)
            collection.add(
                ids=ids_to_add,
                embeddings=embeddings_to_add,
                metadatas=metadata_to_add
                # Optional: documents=texte_der_chunks (wenn du den Text auch in Chroma speichern willst)
            )
            num_added = len(ids_to_add)
            total_chunks_added += num_added
            print(f"    -> Success! Added {num_added} chunks to ChromaDB collection '{collection_name}'.")
            processed_files += 1

        except FileNotFoundError:
             print(f"    -> Error: File not found during processing of {filename} or its .npy counterpart.")
        except Exception as e:
             print(f"    -> An unexpected error occurred processing file pair for '{base_name}': {e}")


print("\nIndexing finished.")
if processed_files > 0:
    print(f"Successfully processed {processed_files} file pairs.")
    print(f"Total chunks added to collection '{collection_name}': {total_chunks_added}")
    print(f"ChromaDB data is stored in: '{chroma_db_path}'")
else:
    print("No files were processed.")