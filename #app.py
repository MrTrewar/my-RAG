import streamlit as st
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import ollama # Oder OpenAI
import time
import traceback
import base64

# --- Konfiguration (aus rag_generate.py übernehmen) ---
chroma_db_path = "chroma_db"
collection_name = "scientific_papers"
embedding_model_name = 'all-MiniLM-L6-v2'
chunks_input_directory = "chunks_output"
# --- WICHTIG: Wähle hier dein gewünschtes OLLAMA-Modell ---
ollama_model_name = "llama3:8b" # Beispiel
ollama_base_url = "http://localhost:11434" # Oder deine OpenAI Konfiguration
retrieval_k = 3
# ---------------------------------------------------------

# --- Caching der Ressourcen für Performance ---
# Lädt Modelle und DB-Client nur einmal, nicht bei jeder Interaktion neu
@st.cache_resource
def load_rag_components():
    print("Loading RAG components...")
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        collection = chroma_client.get_collection(name=collection_name)
        embedding_model = SentenceTransformer(embedding_model_name)
        # Entscheide, ob Ollama oder OpenAI genutzt wird
        llm_client = ollama.Client(host=ollama_base_url) # Oder OpenAI Client
        print("RAG components loaded successfully.")
        # Optional: Hier den Ollama-Check einfügen, wenn gewünscht
        return collection, embedding_model, llm_client
    except Exception as e:
        st.error(f"Fehler beim Laden der RAG-Komponenten: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# --- Hilfsfunktion zum Laden des Chunk-Textes (aus rag_generate.py) ---
# Cache für geladene Chunk-Texte (global oder besser in Session State)
if 'loaded_chunks_cache' not in st.session_state:
    st.session_state.loaded_chunks_cache = {}

def get_chunk_text(metadata):
    document_text = "Error: Could not retrieve text."
    source_file = metadata.get('source_file')
    if source_file:
        json_filename = source_file.replace(".pdf", "_chunks.json")
        chunks_json_path = os.path.join(chunks_input_directory, json_filename)
        try:
            # Cache verwenden
            if chunks_json_path not in st.session_state.loaded_chunks_cache:
                if os.path.exists(chunks_json_path):
                    with open(chunks_json_path, 'r', encoding='utf-8') as f_json:
                        st.session_state.loaded_chunks_cache[chunks_json_path] = json.load(f_json)
                else:
                    st.session_state.loaded_chunks_cache[chunks_json_path] = None # Markieren als nicht gefunden

            chunks_list = st.session_state.loaded_chunks_cache.get(chunks_json_path)

            if chunks_list:
                p_index = metadata.get('paragraph_index')
                c_type = metadata.get('chunk_type')
                found_chunk = None
                # Suche nach dem spezifischen Chunk
                for chunk in chunks_list:
                    chunk_meta = chunk.get('metadata', {})
                    if chunk_meta.get('paragraph_index') == p_index and chunk_meta.get('chunk_type') == c_type:
                         found_chunk = chunk
                         break
                # Fallback (weniger robust)
                if not found_chunk:
                     found_chunk = next((chunk for chunk in chunks_list if chunk.get('metadata') == metadata), None)

                if found_chunk:
                    document_text = found_chunk.get('text', "Error: Text key missing.")
                else:
                    document_text = f"Error: Matching chunk not found (Index: {p_index}, Type: {c_type})."
            elif st.session_state.loaded_chunks_cache[chunks_json_path] is None:
                 document_text = f"Error: Chunks file '{json_filename}' not found."
            else:
                 document_text = f"Error: Chunks file '{json_filename}' could not be loaded or is empty."

        except Exception as e:
            document_text = f"Error loading/processing chunks file '{json_filename}': {e}"
            st.warning(f"Fehler beim Laden von {json_filename}: {e}") # Zeige Warnung in UI
    else:
        document_text = "Error: 'source_file' missing in metadata."
    return document_text

@st.cache_data # Cache das Ergebnis, damit PDF nicht immer neu kodiert wird
def get_pdf_display_link(pdf_filename):
    """Erzeugt einen Base64 Data-URL Link für ein PDF."""
    pdf_path = os.path.join("pdfs_input", pdf_filename)
    if not os.path.exists(pdf_path):
        return None # PDF nicht gefunden

    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        # Erzeugt einen Link, der die Daten direkt im Browser öffnet
        return f"data:application/pdf;base64,{base64_pdf}"
    except Exception as e:
        st.warning(f"Konnte PDF '{pdf_filename}' nicht laden/kodieren: {e}")
        return None

# --- Haupt-RAG-Logik ---
def run_rag_query(query, collection, embedding_model, llm_client):
    results_data = {"answer": "Fehler bei der Verarbeitung.", "sources": []}
    if not query:
        results_data["answer"] = "Bitte gib eine Frage ein."
        return results_data

    try:
        # 1. Retrieval
        start_retrieval = time.time()
        query_embedding = embedding_model.encode(query).tolist()
        chroma_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_k,
            include=['metadatas', 'distances']
        )
        retrieval_time = time.time() - start_retrieval
        print(f"Retrieval took {retrieval_time:.2f} seconds.")

        if not chroma_results or not chroma_results.get('ids') or not chroma_results['ids'][0]:
            results_data["answer"] = "Keine relevanten Textabschnitte gefunden."
            return results_data

        # 2. Kontext aufbereiten & Quelldaten sammeln
        context_parts = []
        source_details = []
        for i in range(len(chroma_results['ids'][0])):
            metadata = chroma_results['metadatas'][0][i]
            distance = chroma_results['distances'][0][i]
            chunk_text = get_chunk_text(metadata)
            source_file = metadata.get('source_file', 'Unbekannte Quelle')

            source_info = {
                "rank": i + 1,
                "distance": distance,
                "source_file": source_file,
                "paragraph_index": metadata.get('paragraph_index', 'N/A'),
                "chunk_type": metadata.get('chunk_type', 'N/A'),
                "text": chunk_text
            }
            source_details.append(source_info)

            if not chunk_text.startswith("Error:"):
                context_parts.append(f"Source: {source_file}, Paragraph {source_info['paragraph_index']}:\n{chunk_text}\n")
            else:
                 context_parts.append(f"Source: {source_file}, Paragraph {source_info['paragraph_index']}:\n[Fehler beim Laden des Textes]\n")


        context_string = "\n---\n".join(context_parts)
        results_data["sources"] = source_details # Speichere Details für die Anzeige

        # 3. Prompt erstellen
        prompt = f"""Kontext:{context_string}
        Frage: {query}

        Anweisung: Beantworte die Frage kurz und prägnant, NUR basierend auf den Informationen im obigen Kontext. Wenn die Antwort nicht explizit im Kontext steht, antworte mit "Die Antwort ist nicht im Kontext enthalten.". Gib keine Quellen im Antworttext an.

        Antwort:"""

# 4. Generation mit LLM
        start_generation = time.time()
        # --- ANPASSEN JE NACH LLM Client (Ollama vs OpenAI) ---
        try:
            # Beispiel für Ollama
            response = llm_client.chat(
                model=ollama_model_name,
                messages=[{"role": "user", "content": prompt}],
                options={'temperature': 0.2, 'num_predict': 200} # Ggf. num_predict anpassen
            )
            generated_answer = response['message']['content'].strip()

            # Beispiel für OpenAI (auskommentiert)
            # response = llm_client.chat.completions.create(
            #     model="gpt-4o", # Oder anderes Modell
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.2,
            #     max_tokens=200
            # )
            # generated_answer = response.choices[0].message.content.strip()

        except Exception as gen_e:
             generated_answer = f"Fehler bei der Antwortgenerierung: {gen_e}"
             st.error(f"Fehler bei LLM-Aufruf: {gen_e}")
             traceback.print_exc()
        # ------------------------------------------------------
        generation_time = time.time() - start_generation
        print(f"Generation took {generation_time:.2f} seconds.")
        results_data["answer"] = generated_answer

    except Exception as e:
        results_data["answer"] = f"Ein Fehler ist aufgetreten: {e}"
        st.error(f"Fehler in run_rag_query: {e}")
        st.error(traceback.print_exc())

    return results_data

# --- Streamlit UI Aufbau ---
st.set_page_config(layout="wide")
st.title("📚 Paper RAG System")

collection, embedding_model, llm_client = load_rag_components()

if collection and embedding_model and llm_client:
    col1, col2 = st.columns([1, 1])

    with col2:
        st.subheader("Frage stellen")
        user_query = st.text_area("Deine Frage:", height=100, key="query_input")
        submit_button = st.button("Antwort generieren")
        st.subheader("Antwort")
        answer_placeholder = st.empty()

    with col1:
        st.subheader("Relevante Textabschnitte (Kontext)")
        sources_placeholder = st.empty()

    if submit_button:
        if user_query:
            with st.spinner("Suche und generiere Antwort..."):
                results = run_rag_query(user_query, collection, embedding_model, llm_client)

            answer_placeholder.markdown(results["answer"])

            with sources_placeholder.container():
                if results["sources"]:
                    for source in results["sources"]:
                        st.markdown(f"**Quelle {source['rank']}: {source['source_file']}** (Index: {source['paragraph_index']}, Typ: {source['chunk_type']}, Distanz: {source['distance']:.4f})")

                        # --- PDF Link hinzufügen ---
                        pdf_link = get_pdf_display_link(source['source_file'])
                        if pdf_link:
                            # Zeige den Link an, der in einem neuen Tab öffnet
                            st.markdown(f'<a href="{pdf_link}" target="_blank">📄 Öffne PDF: {source["source_file"]}</a>', unsafe_allow_html=True)
                        else:
                            st.caption(f"Original PDF ({source['source_file']}) nicht gefunden.")
                        # --------------------------

                        # Zeige den Text-Chunk an
                        st.write(source['text'])
                        st.markdown("---") # Trennlinie
                else:
                    st.write("Keine Quellen gefunden oder Fehler beim Abrufen.")
        else:
            st.warning("Bitte gib eine Frage ein.")

else:
    st.error("Initialisierung der RAG-Komponenten fehlgeschlagen...")