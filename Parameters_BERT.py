"""
Parameters_BERT.py

Uses a BERT-based NER model to extract parameter entities from PDF text, leveraging pre-trained or fine-tuned transformer models.

Features:
- Extracts text from PDFs.
- Loads unit patterns from ECLASS CSV.
- Runs a BERT NER pipeline to identify parameter entities.
- Filters NER results by unit patterns.
"""
import pdfplumber
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 2. Load unit patterns from ECLASS
def load_unit_patterns(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')
    units = set(df['ShortName'].dropna().tolist() + df['SINotation'].dropna().tolist())
    return [u for u in units if isinstance(u, str) and u.strip()]

# 3. Load BERT NER model (pre-trained or fine-tuned)
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")  # or your fine-tuned model
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 4. Extract parameters from text
def extract_parameters(text, unit_patterns):
    results = []
    for sentence in text.split('\n'):
        ner_results = ner_pipeline(sentence)
        for entity in ner_results:
            # Optionally, filter by entity label or check for unit patterns
            if any(unit in entity['word'] for unit in unit_patterns):
                results.append(entity['word'])
    return results

# Example usage
pdf_text = extract_text_from_pdf("metal_jet_s100_English.pdf")
unit_patterns = load_unit_patterns("Datasets/1730150000enUSbasicCSV01/ECLASS15_0_UN_en.csv")
parameters = extract_parameters(pdf_text, unit_patterns)
print(parameters)