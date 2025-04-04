import requests
import os
import time

# --- Konfiguration ---
grobid_url = "http://localhost:8070/api/processFulltextDocument"
pdf_directory = "pdfs_input"  # Ordner mit den zu verarbeitenden PDFs
output_directory = "grobid_output" # Ordner für die resultierenden XML-Dateien
# --------------------

print(f"Checking Grobid service at {grobid_url}...")
try:
    # Einfacher Check, ob Grobid erreichbar ist
    ping_url = "http://localhost:8070/api/isalive"
    ping_response = requests.get(ping_url, timeout=10)
    if ping_response.status_code == 200 and ping_response.text == "true":
        print("Grobid service is alive.")
    else:
        print(f"Grobid service check failed. Status: {ping_response.status_code}, Response: {ping_response.text}")
        print("Bitte stelle sicher, dass der Grobid Docker Container läuft.")
        exit() # Beenden, wenn Grobid nicht läuft
except requests.exceptions.RequestException as e:
    print(f"Could not connect to Grobid service: {e}")
    print("Bitte stelle sicher, dass der Grobid Docker Container läuft und auf Port 8070 erreichbar ist.")
    exit() # Beenden, wenn Grobid nicht erreichbar

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

if not os.path.exists(pdf_directory):
    print(f"Error: Input PDF directory '{pdf_directory}' not found.")
    print("Bitte erstelle den Ordner und lege PDFs hinein.")
    exit()

print(f"Starting PDF processing from '{pdf_directory}'...")

# Gehe durch alle Dateien im PDF-Verzeichnis
for filename in os.listdir(pdf_directory):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        # Erstelle einen eindeutigen Namen für die XML-Datei
        base_filename = os.path.splitext(filename)[0]
        output_xml_path = os.path.join(output_directory, f"{base_filename}_grobid.xml")

        print(f"  Processing: {filename} ...")

        try:
            with open(pdf_path, 'rb') as f:
                files = {
                    # Wichtig: Der Dateiname im Tupel muss nicht der Originalname sein,
                    # aber es ist oft hilfreich.
                    'input': (filename, f, 'application/pdf')
                    }
                # Parameter für die Grobid API
                params = {
                    'consolidateHeader': '1', # Versucht Metadaten zu verbessern
                    'segmentSentences': 'true' # Segmentiert den Text in Sätze (<s> Tags)
                }

                # Sende die Anfrage an Grobid
                start_time = time.time()
                response = requests.post(grobid_url, files=files, data=params, timeout=300) # Timeout erhöhen (Sekunden)
                end_time = time.time()

            # Überprüfe, ob die Anfrage erfolgreich war (Status Code 200)
            response.raise_for_status()

            # Speichere die XML-Antwort
            with open(output_xml_path, 'w', encoding='utf-8') as out_f:
                out_f.write(response.text)

            processing_time = end_time - start_time
            print(f"    -> Success! Saved XML to '{output_xml_path}' (Time: {processing_time:.2f}s)")

        except requests.exceptions.Timeout:
             print(f"    -> Error: Request timed out for {filename}. Grobid braucht möglicherweise länger oder die PDF ist sehr komplex.")
        except requests.exceptions.RequestException as e:
            print(f"    -> Error processing {filename}: {e}")
            # Optional: Detailliertere Fehlermeldung ausgeben, falls vorhanden
            try:
                error_detail = response.text
                print(f"       Grobid Response: {error_detail[:500]}...") # Zeige Anfang der Antwort
            except:
                pass # Falls keine Antwort verfügbar
        except Exception as e:
             print(f"    -> An unexpected error occurred for {filename}: {e}")

print("\nGrobid processing finished.")
print(f"XML outputs are in the '{output_directory}' folder.")