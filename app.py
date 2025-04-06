# --- Imports ---
from flask import Flask, render_template, request, url_for
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
# import ollama # <-- Entfernen oder auskommentieren
import google.generativeai as genai # <-- NEU: Google AI importieren
from dotenv import load_dotenv # <-- NEU: Für .env Datei
import time
import traceback
import base64

load_dotenv()

# --- Konfiguration (wie zuvor) ---
chroma_db_path = "chroma_db"
collection_name = "scientific_papers"
embedding_model_name = 'all-MiniLM-L6-v2'
chunks_input_directory = "chunks_output"
pdfs_input_directory = "pdfs_input"
GEMINI_MODEL_NAME = "Gemini 2.0 Flash-Lite"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
##ollama_base_url = "http://localhost:11434"
retrieval_k = 3
# ---------------------------------

# --- Flask App Initialisierung ---
app = Flask(__name__)
# Optional: Secret Key für Flask Sessions (falls benötigt, hier nicht direkt)
# app.secret_key = 'ein_sehr_geheimer_schluessel'
# ---------------------------------

# --- RAG Komponenten Laden (Global beim Start) ---
# HINWEIS: Das Laden großer Modelle hier kann den Start verzögern.
# In Produktionsumgebungen gibt es bessere Muster (z.B. Application Factories).
print("Loading RAG components...")
RAG_COMPONENTS_LOADED = False
rag_collection = None
rag_embedding_model = None
rag_gemini_model = None # <-- NEU: Variable für Gemini Modell
# --- NEU: Prüfe API Key und konfiguriere Gemini ---
if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY) # <-- NEU: Google API konfigurieren
        print("Google AI SDK configured.")

        # Lade andere Komponenten
        rag_collection = chromadb.PersistentClient(path=chroma_db_path).get_collection(name=collection_name)
        rag_embedding_model = SentenceTransformer(embedding_model_name)

        # --- NEU: Initialisiere das Gemini Modell ---
        rag_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Gemini model '{GEMINI_MODEL_NAME}' loaded.")
        # ------------------------------------------

        print("RAG components loaded successfully.")
        RAG_COMPONENTS_LOADED = True

    except Exception as e:
        print(f"FATAL ERROR: Could not load RAG components: {e}")
        print(traceback.format_exc())
# --------------------------------------------------

# --- Chunk Cache (einfache globale Variable für Flask) ---
loaded_chunks_cache = {}
# -------------------------------------------------------

# --- Hilfsfunktion get_chunk_text (angepasst für Flask) ---
def get_chunk_text(metadata):
    global loaded_chunks_cache # Zugriff auf globale Cache-Variable
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
                # Finde den Chunk (Logik wie zuvor)
                found_chunk = None
                for chunk in chunks_list:
                     chunk_meta = chunk.get('metadata', {})
                     if chunk_meta.get('paragraph_index') == p_index and chunk_meta.get('chunk_type') == c_type:
                         found_chunk = chunk
                         break
                if not found_chunk:
                     found_chunk = next((c for c in chunks_list if c.get('metadata') == metadata), None)

                if found_chunk:
                    document_text = found_chunk.get('text', "Error: Text key missing.")
                else:
                    document_text = f"Error: Matching chunk not found."
            elif loaded_chunks_cache.get(chunks_json_path) is None:
                 document_text = f"Error: Chunks file '{json_filename}' not found."
            else:
                 document_text = f"Error: Chunks file empty or invalid."
        except Exception as e:
            document_text = f"Error loading/processing chunks file: {e}"
            print(f"Warning: Error loading {json_filename}: {e}") # Log zum Terminal
    else:
        document_text = "Error: 'source_file' missing."
    return document_text
# ----------------------------------------------------------

# --- Hilfsfunktion get_pdf_display_link (angepasst für Flask) ---
# Caching hier nicht so einfach wie mit @st.cache_data, erstmal ohne
def get_pdf_display_link(pdf_filename):
    pdf_path = os.path.join(pdfs_input_directory, pdf_filename)
    if not os.path.exists(pdf_path):
        return None
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        return f"data:application/pdf;base64,{base64_pdf}"
    except Exception as e:
        print(f"Warning: Could not load/encode PDF '{pdf_filename}': {e}") # Log zum Terminal
        return None
# -------------------------------------------------------------

# --- Haupt-RAG-Logik (angepasst für Flask) ---
def run_rag_query(query):
    # Greife auf die global geladenen Komponenten zu
    if not RAG_COMPONENTS_LOADED:
         return {"answer": "Fehler: RAG-Komponenten nicht geladen.", "sources": []}

    collection = rag_collection
    embedding_model = rag_embedding_model
    gemini_model = rag_gemini_model # <-- Geändert

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
        print(f"Retrieval took {time.time() - start_retrieval:.2f} seconds.")

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
            pdf_link = get_pdf_display_link(source_file) # Erzeuge PDF Link

            source_info = {
                "rank": i + 1,
                "distance": distance,
                "source_file": source_file,
                "paragraph_index": metadata.get('paragraph_index', 'N/A'),
                "chunk_type": metadata.get('chunk_type', 'N/A'),
                "text": chunk_text,
                "pdf_link": pdf_link # Füge Link hinzu
            }
            source_details.append(source_info)

            # Kontext für LLM
            if not chunk_text.startswith("Error:"):
                context_parts.append(f"Source: {source_file}, Paragraph {source_info['paragraph_index']}:\n{chunk_text}\n")
            else:
                 context_parts.append(f"Source: {source_file}, Paragraph {source_info['paragraph_index']}:\n[Fehler beim Laden des Textes]\n")

        context_string = "\n---\n".join(context_parts)
        results_data["sources"] = source_details

        # 3. Prompt erstellen (wie zuvor)
        prompt = f"""Kontext: ...""" # Dein Prompt hier

        # 4. Generation mit LLM (wie zuvor)
        # --- 4. Generation mit Gemini (ANGEPASST) ---
        start_generation = time.time()
        generated_answer = f"Fehler bei der Antwortgenerierung mit {GEMINI_MODEL_NAME}." # Default Fehler
        try:
            # Sicherheitseinstellungen (Optional, aber empfohlen)
            # Verhindert potenziell unsichere Antworten, kann aber manchmal harmlose Antworten blockieren
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            response = gemini_model.generate_content(
                prompt,
                safety_settings=safety_settings
                # Optional: generation_config = genai.types.GenerationConfig(...)
            )

            # Antwort extrahieren (mit Fehlerprüfung für blockierte Inhalte)
            try:
                 generated_answer = response.text.strip()
            except ValueError:
                 # Wahrscheinlich wurde die Antwort wegen Sicherheitseinstellungen blockiert
                 print(f"Warning: Gemini response blocked. Feedback: {response.prompt_feedback}")
                 generated_answer = "Die Antwort wurde aufgrund von Sicherheitseinstellungen blockiert."
                 # Alternativ: Details aus response.candidates prüfen, falls vorhanden
                 # if response.candidates: generated_answer = f"Blocked: {response.candidates[0].finish_reason} / {response.candidates[0].safety_ratings}"

        except Exception as gen_e:
             print(f"Error calling Gemini API: {gen_e}")
             traceback.print_exc()
             # Behalte den Default-Fehler oder spezifiziere
             generated_answer = f"Fehler bei der Kommunikation mit der Gemini API: {gen_e}"
        # ------------------------------------------

        generation_time = time.time() - start_generation
        print(f"Generation took {generation_time:.2f} seconds.")
        results_data["answer"] = generated_answer

    except Exception as e:
        results_data["answer"] = f"Ein Fehler ist aufgetreten: {e}"
        print(f"Error in run_rag_query: {e}")
        print(traceback.format_exc())

    return results_data
# ------------------------------------------------

# --- Flask Routen ---
@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    results = {"answer": "", "sources": []} # Standardmäßig leer

    if request.method == 'POST':
        # Formular wurde gesendet
        query = request.form.get('query', '') # Hole Query aus Formularfeld
        if query and RAG_COMPONENTS_LOADED:
            print(f"Received query: {query}")
            results = run_rag_query(query) # Führe RAG-Logik aus
        elif not RAG_COMPONENTS_LOADED:
             results = {"answer": "System-Initialisierungsfehler. Bitte Logs prüfen.", "sources": []}


    # Rendere die HTML-Seite und übergebe Variablen
    # 'last_query' wird verwendet, um die Frage im Feld anzuzeigen
    return render_template('index.html',
                           last_query=query,
                           answer=results["answer"],
                           sources=results["sources"])

# Optional: Route zum direkten Servieren von PDFs (Alternative zu Base64)
# @app.route('/pdf/<path:filename>')
# def serve_pdf(filename):
#     try:
#         return send_from_directory(pdfs_input_directory, filename, as_attachment=False)
#     except FileNotFoundError:
#         abort(404)
# --------------------

# --- App Start ---
if __name__ == '__main__':
    # debug=True startet den Development Server neu bei Änderungen
    # NICHT in Produktion verwenden!
    app.run(debug=True, port=5001) # Port ändern, falls 5000 belegt ist
# ---------------