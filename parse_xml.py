import os
import json # Importiere das JSON-Modul
from lxml import etree

# --- Konfiguration ---
xml_input_directory = "grobid_output" # Ordner mit den Grobid XML-Dateien
chunks_output_directory = "chunks_output" # Neuer Ordner für die Chunk-JSON-Dateien
# --------------------

print(f"Starting XML parsing and chunking from '{xml_input_directory}'...")

# Erstelle das Ausgabe-Verzeichnis für Chunks, falls es nicht existiert
if not os.path.exists(chunks_output_directory):
    os.makedirs(chunks_output_directory)
    print(f"Created chunks output directory: {chunks_output_directory}")

if not os.path.exists(xml_input_directory):
    print(f"Error: XML input directory '{xml_input_directory}' not found.")
    exit()

# Gehe durch alle Dateien im XML-Verzeichnis
for filename in os.listdir(xml_input_directory):
    if filename.lower().endswith(".xml"):
        xml_path = os.path.join(xml_input_directory, filename)
        # Erstelle einen Basisnamen für die Output-JSON-Datei (entferne _grobid.xml)
        base_output_name = filename.replace("_grobid.xml", "").replace(".xml", "")
        json_output_path = os.path.join(chunks_output_directory, f"{base_output_name}_chunks.json")

        print(f"  Processing: {filename} ...")

        try:
            # Lade die XML-Datei
            tree = etree.parse(xml_path)
            root = tree.getroot()
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'} # Namespace

            # --- Extraktion ---
            title_elements = root.xpath('//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title', namespaces=ns)
            title = title_elements[0].xpath('string()').strip() if title_elements else "Title not found"

            abstract_elements = root.xpath('//tei:teiHeader/tei:profileDesc/tei:abstract/tei:p', namespaces=ns)
            abstract = abstract_elements[0].xpath('string()').strip() if abstract_elements else "" # Leerer String, falls nicht gefunden

            # --- Chunking (Jeder Paragraph wird ein Chunk) ---
            file_chunks = [] # Liste für die Chunks dieser Datei
            paragraph_elements = root.xpath('//tei:text/tei:body//tei:p', namespaces=ns)

            # Optional: Füge Abstract als ersten Chunk hinzu, wenn vorhanden
            if abstract:
                 chunk_metadata = {
                        "source_file": filename.replace("_grobid.xml", ".pdf"), # Versuche, den PDF-Namen zu rekonstruieren
                        "title": title,
                        "chunk_type": "abstract" # Spezifischer Typ
                    }
                 file_chunks.append({
                     "text": ' '.join(abstract.split()), # Bereinigter Abstract-Text
                     "metadata": chunk_metadata
                 })

            # Füge Paragraphen als Chunks hinzu
            for i, p_element in enumerate(paragraph_elements):
                para_text = p_element.xpath('string()').strip()
                cleaned_text = ' '.join(para_text.split()) # Bereinigter Text

                if cleaned_text: # Nur nicht-leere Paragraphen hinzufügen
                    # Erstelle Metadaten für diesen Chunk
                    chunk_metadata = {
                        "source_file": filename.replace("_grobid.xml", ".pdf"), # Versuche, den PDF-Namen zu rekonstruieren
                        "title": title,
                        "chunk_type": "paragraph",
                        "paragraph_index": i # 0-basierter Index des Paragraphen
                    }
                    # Erstelle den Chunk (Dictionary) und füge ihn zur Liste hinzu
                    file_chunks.append({
                        "text": cleaned_text,
                        "metadata": chunk_metadata
                    })

            # --- Speichere die Chunks als JSON-Datei ---
            if file_chunks: # Nur speichern, wenn Chunks gefunden wurden
                with open(json_output_path, 'w', encoding='utf-8') as f_out:
                    # json.dump schreibt die Python-Liste als JSON in die Datei
                    # indent=2 sorgt für schöne Formatierung (Einrückung)
                    # ensure_ascii=False erlaubt Sonderzeichen direkt (wichtig für Umlaute etc.)
                    json.dump(file_chunks, f_out, ensure_ascii=False, indent=2)
                print(f"    -> Success! Found {len(file_chunks)} chunks. Saved chunks to '{json_output_path}'")
            else:
                print(f"    -> Warning: No text paragraphs found in the body of {filename}. No chunks saved.")


        except etree.XMLSyntaxError as e:
            print(f"    -> Error: Could not parse XML file {filename}. Error: {e}")
        except Exception as e:
             print(f"    -> An unexpected error occurred processing {filename}: {e}")


print("\nXML parsing and chunking finished.")
print(f"Chunk JSON files are in the '{chunks_output_directory}' folder.")