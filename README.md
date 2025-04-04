Workflow: Neue PDF hinzufügen

PDF kopieren: Neue PDF-Datei in den Ordner pdfs_input/ legen.

(Check): GROBID läuft? (Nur falls process_pdfs.py es braucht).

Terminal öffnen & venv aktivieren:

cd /pfad/zu/deinem/projekt

source venv/bin/activate (Mac/Linux) ODER .\venv\Scripts\activate (Windows)

Verarbeiten: python process_pdfs.py

Parsen: python parse_xml.py

Indizieren: python index_data.py

Abfragen: python rag_generate.py starten (oder neu starten, falls es lief).






# RAG-Pipeline für wissenschaftliche Paper

Dieses Projekt implementiert eine Retrieval-Augmented Generation (RAG) Pipeline, um Fragen zu wissenschaftlichen Artikeln (im PDF-Format) mithilfe von Ollama, ChromaDB und Sentence Transformers zu beantworten. Es nutzt wahrscheinlich [GROBID](https://github.com/kermitt2/grobid) zur Extraktion von strukturiertem Text aus PDFs.

## Voraussetzungen

Bevor du beginnst, stelle sicher, dass die folgende Software installiert ist:

*   **Python:** Version 3.9 oder höher empfohlen.
*   **Git:** Zum Klonen des Repositories.
*   **Ollama:** Installiert und der Ollama-Server muss laufen. Stelle sicher, dass das benötigte Sprachmodell heruntergeladen ist (siehe Setup). [Ollama Website](https://ollama.com/)
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
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ollama Modell herunterladen:**
    Lade das benötigte Sprachmodell für Ollama herunter (das Standardmodell ist in `rag_generate.py` konfiguriert, z.B. `qwen:0.5b`).
    ```bash
    ollama pull qwen:0.5b
    ```
    *(Ersetze `qwen:0.5b` falls ein anderes Modell in der Konfiguration verwendet wird)*

## Datenvorbereitung (Neue PDFs hinzufügen)

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

## Benutzung (Fragen stellen)

1.  **Voraussetzungen prüfen:**
    *   Stelle sicher, dass deine virtuelle Umgebung (`venv`) aktiviert ist.
    *   Stelle sicher, dass der Ollama-Server läuft.
    *   Stelle sicher, dass die Datenvorbereitung (siehe oben) mindestens einmal durchlaufen wurde und die `chroma_db/` existiert.
2.  **Abfrage-Skript starten:**
    ```bash
    # Stelle sicher, dass venv aktiviert ist
    python rag_generate.py
    ```
3.  Folge den Anweisungen im Terminal, um Fragen zu stellen. Gib 'exit' ein, um das Programm zu beenden.

## Konfiguration

Wichtige Einstellungen können am Anfang der Datei `rag_generate.py` angepasst werden:

*   `chroma_db_path`: Pfad zur ChromaDB-Datenbank.
*   `collection_name`: Name der ChromaDB-Kollektion.
*   `embedding_model_name`: Name des Sentence Transformer Modells für Embeddings.
*   `chunks_input_directory`: Verzeichnis mit den Chunk-JSON-Dateien.
*   `ollama_model_name`: Name des Ollama-Modells für die Textgenerierung.
*   `ollama_base_url`: URL des laufenden Ollama-Servers.
*   `retrieval_k`: Anzahl der Chunks, die als Kontext abgerufen werden sollen.

## Projektstruktur
├── chroma_db/ # ChromaDB Vektordatenbank
├── chunks_output/ # Extrahierte Text-Chunks (JSON)
├── embeddings_output/ # (Optional, Zweck unklar aus Bild)
├── grobid_output/ # Von GROBID verarbeitete XML-Dateien
├── pdfs_input/ # Eingabe-PDF-Dateien
├── venv/ # Virtuelle Python-Umgebung
├── .env # (Optional, für Umgebungsvariablen)
├── .gitignore # Git-Ignorierdatei
├── embed_chunks.py # (Vermutlich Teil von index_data.py oder separate Einbettung?)
├── index_data.py # Skript zum Indizieren von Chunks in ChromaDB
├── parse_xml.py # Skript zum Parsen von GROBID-XML und Erstellen von Chunks
├── process_pdfs.py # Skript zur Verarbeitung von PDFs (wahrscheinlich mit GROBID)
├── query_data.py # (Vielleicht ältere Version oder Testskript für reine Abfrage?)
├── rag_generate.py # Hauptskript für die RAG-Abfrage
├── requirements.txt # Python-Abhängigkeiten
├── test_hallo.txt # Testdatei
├── test_ollama.py # Testskript für Ollama-Verbindung
└── README.md # Diese Datei


## Troubleshooting

*   **Fehler bei Ollama-Verbindung:** Stelle sicher, dass der Ollama-Dienst läuft und unter der in `rag_generate.py` konfigurierten `ollama_base_url` erreichbar ist.
*   **Modell nicht gefunden:** Stelle sicher, dass das in `ollama_model_name` angegebene Modell mit `ollama pull <modellname>` heruntergeladen wurde.
*   **Python-Fehler (`ModuleNotFoundError`, etc.):** Stelle sicher, dass deine virtuelle Umgebung (`venv`) aktiviert ist und alle Abhängigkeiten mit `pip install -r requirements.txt` installiert wurden.
*   **ChromaDB-Fehler:** Überprüfe den Pfad `chroma_db_path` und stelle sicher, dass die Datenbank existiert und gültig ist (nachdem `index_data.py` gelaufen ist).

## Lizenz

(Optional: Füge hier deine Lizenzinformationen hinzu, z.B. MIT License)
