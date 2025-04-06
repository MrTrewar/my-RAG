Workflow: Neue PDF hinzufügen (mit Flask App)

PDF kopieren: Neue PDF-Datei in den Ordner pdfs_input/ legen.

(Check): GROBID läuft? (Nur falls process_pdfs.py es braucht).

Terminal öffnen & venv aktivieren:

cd /pfad/zu/deinem/projekt

source venv/bin/activate (Mac/Linux) ODER .\venv\Scripts\activate (Windows)

Verarbeiten: python process_pdfs.py

Parsen: python parse_xml.py

Indizieren: python index_data.py

Web App starten/neu starten: python app.py starten (oder neu starten, falls sie bereits lief, oft Strg+C zum Beenden und dann neu starten).

Abfragen: App im Browser öffnen (z.B. http://127.0.0.1:5001) und Fragen stellen.





 RAG-Pipeline für wissenschaftliche Paper (Flask Web App)

Dieses Projekt implementiert eine Retrieval-Augmented Generation (RAG) Pipeline als Webanwendung, um Fragen zu wissenschaftlichen Artikeln (im PDF-Format) mithilfe von Ollama, ChromaDB und Sentence Transformers zu beantworten. Die Webanwendung basiert auf **Flask**. Das Projekt nutzt wahrscheinlich [GROBID](https://github.com/kermitt2/grobid) zur Extraktion von strukturiertem Text aus PDFs für die Datenvorbereitung.

## Voraussetzungen

Bevor du beginnst, stelle sicher, dass die folgende Software installiert ist:

*   **Python:** Version 3.9 oder höher empfohlen.
*   **Git:** Zum Klonen des Repositories.
*   **Ollama:** Installiert und der Ollama-Server muss laufen. Stelle sicher, dass das benötigte Sprachmodell heruntergeladen ist (siehe Setup). [Ollama Website](https://ollama.com/)
*   **Flask:** Wird über `requirements.txt` installiert.
*   **GROBID:** (Optional, aber wahrscheinlich für `process_pdfs.py` benötigt) Ein laufender GROBID-Dienst. Folge der [GROBID Installationsanleitung](https://grobid.readthedocs.io/en/latest/Install-Grobid/). Der Standard-Port ist `8070`.

## Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Virtuelle Umgebung (venv) erstellen und aktivieren:**
    *Es wird dringend empfohlen, eine virtuelle Umgebung zu verwenden, um Abhängigkeiten zu isolieren.*

    *   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *Hinweis: Du musst die virtuelle Umgebung (`source venv/bin/activate` oder `.\venv\Scripts\activate`) in deinem Terminal jedes Mal aktivieren, wenn du an diesem Projekt arbeitest.*

3.  **Abhängigkeiten installieren:**
    *Stelle sicher, dass deine virtuelle Umgebung aktiviert ist.*
    *(**Wichtig:** Stelle sicher, dass `Flask` in deiner `requirements.txt` enthalten ist. Falls nicht, füge es hinzu (`pip install Flask`) und aktualisiere die Datei: `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ollama Modell herunterladen:**
    Lade das benötigte Sprachmodell für Ollama herunter (das Standardmodell ist in `app.py` konfiguriert, z.B. `llama3:8b`).
    ```bash
    ollama pull llama3:8b
    ```
    *(Ersetze `llama3:8b` falls ein anderes Modell in der Konfiguration verwendet wird)*

## Datenvorbereitung (Neue PDFs hinzufügen)

**Diese Schritte sind identisch wie zuvor und müssen ausgeführt werden, bevor die Web-App nützliche Ergebnisse liefern kann.**

Führe die folgenden Schritte aus, um neue PDF-Dokumente für die Abfrage verfügbar zu machen:

1.  **PDFs kopieren:** Lege deine PDF-Dateien in den Ordner `pdfs_input/`.
2.  **GROBID starten:** Stelle sicher, dass dein GROBID-Dienst läuft (falls `process_pdfs.py` ihn benötigt).
3.  **PDFs verarbeiten:** Führe das Skript aus, um die PDFs zu verarbeiten (z.B. Text mit GROBID extrahieren und als XML speichern).
    ```bash
    # Stelle sicher, dass venv aktiviert ist
    python process_pdfs.py
    ```
    *(Erzeugt wahrscheinlich XML-Dateien in `grobid_output/`)*
4.  **XML parsen & Chunks erstellen:** Wandle die strukturierten Daten (XML) in Text-Chunks um.
    ```bash
    # Stelle sicher, dass venv aktiviert ist
    python parse_xml.py
    ```
    *(Erzeugt wahrscheinlich JSON-Dateien mit Chunks in `chunks_output/`)*
5.  **Daten indizieren:** Erstelle Embeddings für die neuen Chunks und füge sie zur Vektordatenbank hinzu.
    ```bash
    # Stelle sicher, dass venv aktiviert ist
    python index_data.py
    ```
    *(Aktualisiert die ChromaDB in `chroma_db/`)*

## Benutzung der Web App (Fragen stellen)

1.  **Voraussetzungen prüfen:**
    *   Stelle sicher, dass deine virtuelle Umgebung (`venv`) aktiviert ist.
    *   Stelle sicher, dass der Ollama-Server läuft.
    *   Stelle sicher, dass die Datenvorbereitung (siehe oben) mindestens einmal durchlaufen wurde und die `chroma_db/` existiert und die RAG-Komponenten beim Start von `app.py` erfolgreich geladen werden können (siehe Terminal-Ausgabe beim Start).
2.  **Flask Web App starten:**
    ```bash
    # Stelle sicher, dass venv aktiviert ist
    python app.py
    ```
    *(Das Skript startet einen lokalen Webserver, normalerweise auf Port 5001 oder 5000. Achte auf die Ausgabe im Terminal.)*
3.  **App im Browser öffnen:** Öffne deinen Webbrowser und gehe zur angezeigten Adresse, z.B. `http://127.0.0.1:5001`.
4.  **Fragen stellen:** Gib deine Frage in das Textfeld in der rechten Spalte ein und klicke auf "Antwort generieren". Die Antwort erscheint rechts, die relevanten Kontext-Abschnitte links.

## Konfiguration

Wichtige Einstellungen können am Anfang der **Flask-Anwendungsdatei `app.py`** angepasst werden:

*   `chroma_db_path`: Pfad zur ChromaDB-Datenbank.
*   `collection_name`: Name der ChromaDB-Kollektion.
*   `embedding_model_name`: Name des Sentence Transformer Modells für Embeddings.
*   `chunks_input_directory`: Verzeichnis mit den Chunk-JSON-Dateien.
*   `pdfs_input_directory`: Verzeichnis mit den Original-PDF-Dateien (für Links).
*   `ollama_model_name`: Name des Ollama-Modells für die Textgenerierung.
*   `ollama_base_url`: URL des laufenden Ollama-Servers.
*   `retrieval_k`: Anzahl der Chunks, die als Kontext abgerufen werden sollen.
*   *(Optional: Flask-spezifische Einstellungen wie Port in `app.run()`)*

## Projektstruktur (Aktualisiert für Flask)
Use code with caution.
Markdown
├── app.py # Haupt-Flask-Anwendungsdatei << NEU/GEÄNDERT
├── templates/ # Ordner für HTML-Templates << NEU
│ └── index.html # Haupt-HTML-Seite << NEU
├── static/ # Ordner für CSS, JS, Bilder << NEU
│ └── style.css # CSS-Datei << NEU
├── chroma_db/ # ChromaDB Vektordatenbank
├── chunks_output/ # Extrahierte Text-Chunks (JSON)
├── grobid_output/ # Von GROBID verarbeitete XML-Dateien
├── pdfs_input/ # Eingabe-PDF-Dateien
├── venv/ # Virtuelle Python-Umgebung
├── .gitignore # Git-Ignorierdatei
├── index_data.py # Skript zum Indizieren von Chunks in ChromaDB
├── parse_xml.py # Skript zum Parsen von GROBID-XML und Erstellen von Chunks
├── process_pdfs.py # Skript zur Verarbeitung von PDFs (wahrscheinlich mit GROBID)
├── requirements.txt # Python-Abhängigkeiten
└── README.md # Diese Datei

--- Potenzielle Überbleibsel/Diagnose-Skripte ---
├── rag_generate.py # Altes CLI Skript (ersetzt durch app.py)
├── test_ollama.py # Testskript für Ollama-Verbindung
├── embed_chunks.py # (Rolle unklar, evtl. veraltet/integriert?)
├── query_data.py # (Rolle unklar, evtl. veraltet?)
├── test_hallo.txt # (Wahrscheinlich löschbar)
├── .env # (Sollte in .gitignore sein)
├── embeddings_output/ # (Sollte in .gitignore sein)
## Troubleshooting

*   **Flask App startet nicht / Fehler 500:** Prüfe die Terminal-Ausgabe von `python app.py` auf Fehlermeldungen. Häufige Ursachen: Fehler beim Laden der RAG-Komponenten, Syntaxfehler, Template nicht gefunden (`templates/index.html` muss existieren).
*   **`TemplateNotFound` Fehler:** Stelle sicher, dass deine `index.html` Datei im `templates` Ordner liegt.
*   **Styling nicht angewendet:** Stelle sicher, dass die `style.css` Datei im `static` Ordner liegt und der Link im HTML korrekt ist (`{{ url_for('static', ...) }}`). Leere den Browser-Cache.
*   **Fehler bei Ollama-Verbindung:** Stelle sicher, dass der Ollama-Dienst läuft und unter der in `app.py` konfigurierten `ollama_base_url` erreichbar ist.
*   **Modell nicht gefunden:** Stelle sicher, dass das in `ollama_model_name` angegebene Modell mit `ollama pull <modellname>` heruntergeladen wurde.
*   **Python-Fehler (`ModuleNotFoundError`, etc.):** Stelle sicher, dass deine virtuelle Umgebung (`venv`) aktiviert ist und alle Abhängigkeiten mit `pip install -r requirements.txt` installiert wurden (insbesondere `Flask`).
*   **ChromaDB-Fehler:** Überprüfe den Pfad `chroma_db_path` und stelle sicher, dass die Datenbank existiert und gültig ist (nachdem `index_data.py` gelaufen ist).

## Lizenz

(Optional: Füge hier deine Lizenzinformationen hinzu, z.B. MIT License)