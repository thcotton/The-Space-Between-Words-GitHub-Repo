"""
Text Extract:
"""

import csv
import os
from tqdm import tqdm
import pdfplumber

input_csv = r"C:\Users\toric\OneDrive\Desktop\Gamete Papers\Gamete Library Export Zotero.csv"
output_csv = r"C:\Users\toric\OneDrive\Desktop\Gamete Papers\The Great Gamete Dataset.csv"

rows = []

# Load Zotero metadata
with open(input_csv, newline='', encoding='utf-8') as infile:
    reader = list(csv.DictReader(infile))

    for row in tqdm(reader, desc="Processing PDFs"):
        file_path = row.get("File Attachments", "").strip()
        text = ""

        if file_path and os.path.exists(file_path):
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as e:
                print(f" Could not read PDF: {file_path}\nError: {e}")
        
        row["FullText"] = text
        rows.append(row)

# Write new CSV with extracted text
with open(output_csv, "w", newline='', encoding='utf-8') as outfile:
    fieldnames = rows[0].keys()
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Full text added to CSV.")