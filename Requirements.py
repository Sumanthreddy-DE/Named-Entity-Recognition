import pdfplumber
import re
import csv
import pandas as pd
from pathlib import Path


def load_eclass_keywords():
    """Load keywords from eCl@ss dataset"""
    dataset_path = Path("Datasets/1730150000enUSbasicCSV01/ECLASS15_0_KWSY_en.csv")
    try:
        # Read the CSV file
        df = pd.read_csv(dataset_path, encoding='utf-8')
        # Extract keywords and their synonyms
        keywords = set()
        for col in df.columns:
            if 'keyword' in col.lower() or 'synonym' in col.lower():
                keywords.update(df[col].dropna().str.lower().tolist())
        return keywords
    except Exception as e:
        print(f"Error loading eCl@ss dataset: {e}")
        # Fallback to basic keywords if dataset loading fails
        return {'must', 'should', 'required', 'recommended', 'only', 'not'}


# 1. Extract all text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:50]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_requirements(text, keywords):
    # Split the text into sentences 
    sentences = re.split(r'[.\n]', text)
    requirements = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if any keyword is present in the sentence
        if any(keyword in sentence.lower() for keyword in keywords):
            # Additional filtering to ensure it's a requirement
            if len(sentence.split()) >= 3:  # Avoid very short phrases
                requirements.append(sentence)
    
    return requirements

# 6. Main function to run all steps
def run_information_extraction(pdf_path):
    # Load keywords from eCl@ss dataset
    keywords = load_eclass_keywords()
    
    # Extract text and requirements
    text = extract_text_from_pdf(pdf_path)
    requirements = extract_requirements(text, keywords)
    
    print("=== REQUIREMENTS ===")
    print(f"Found {len(requirements)} requirements using eCl@ss dataset")
    for req in requirements[:10]:  # Show first 10 requirements
        print("-", req)


# Example usage:
if __name__ == "__main__":
    run_information_extraction("metal_jet_s100_English.pdf")