"""
param&safety.py

Extracts technical parameters (with units) and safety-related sentences from PDF documents using pattern matching and keyword mining.

Features:
- Extracts text from PDFs.
- Loads unit patterns from ECLASS CSV files.
- Extracts parameters (e.g., temperature, pressure) using regex.
- Extracts safety warnings/indicators using keyword mining from ECLASS CSVs.
"""
import pdfplumber
import re
import csv


# 1. Extract all text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:10]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text



# 2.1: get units from excel file
def get_unit_patterns():
    patterns = []
    unit_info = {}  # Dictionary to store unit information
    with open('Datasets/1730150000enUSbasicCSV01/ECLASS15_0_UN_en.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            # Get the unit information
            short_name = row.get('ShortName', '').strip()
            si_notation = row.get('SINotation', '').strip()
            
            # Create pattern for the unit
            if short_name:
                # Escape special characters and create pattern
                escaped_name = re.escape(short_name)
                pattern = rf"(\d+\s?{escaped_name})"
                patterns.append(pattern)
                # Store unit info with pattern as key
                unit_info[pattern] = {
                    'unit': short_name
                }
            
            if si_notation:
                # Escape special characters and create pattern
                escaped_notation = re.escape(si_notation)
                pattern = rf"(\d+\s?{escaped_notation})"
                patterns.append(pattern)
                # Store unit info with pattern as key
                unit_info[pattern] = {
                    'unit': si_notation
                }
    
    return list(set(patterns)), unit_info  # Return both patterns and unit info


#text mining
# 2.2: Extract technical parameters (temperature, pressure, airflow, weight, etc.)
def extract_parameters(text):
    sentences = re.split(r'[.\n]', text)
    patterns, unit_info = get_unit_patterns()
    params = []
    for pat in patterns:
        matches = re.findall(pat, text)
        for match in matches:
            # Get the unit information for this pattern
            info = unit_info[pat]
            # Create parameter string with just the value and unit
            param_str = f"{match}"
            params.append(param_str)
    return list(set(params))

#Natural Language Processing (NLP)
# 3.1: Extract safety indicators 
def get_safety_keywords():
    safety_keywords = set()
    
    with open('Datasets/1730150000enUSbasicCSV01/ECLASS15_0_CC_en.csv', 'r', encoding='utf-8') as f: #UTF-8 encoding to handle special characters.
            reader = csv.DictReader(f, delimiter=';') #reads each row of the CSV file as a dictionary.
            for row in reader:
                # Look for safety-related classifications
                for column in row:
                    value = row.get(column, '').lower()
                    if any(term in value for term in ['safety', 'warning', 'caution', 'danger', 
                                                    'hazard', 'risk', 'protect', 'secure']):
                        safety_keywords.add(value)

    with open('Datasets/1730150000enUSbasicCSV01/ECLASS15_0_PR_en.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Look for safety-related properties
                for column in row:
                    value = row.get(column, '').lower()
                    if any(term in value for term in ['safety', 'warning', 'caution', 'danger', 
                                                    'hazard', 'risk', 'protect', 'secure']):
                        safety_keywords.add(value)
    
    
    
    with open('Datasets/1730150000enUSbasicCSV01/ECLASS15_0_PR_VA_restricted_en.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Look for safety-related restricted properties
                for column in row:
                    value = row.get(column, '').lower()
                    if any(term in value for term in ['safety', 'warning', 'caution', 'danger', 
                                                    'hazard', 'risk', 'protect', 'secure']):
                        safety_keywords.add(value)
    
    return safety_keywords

#3.2 Extract safety parameters
def extract_safety(text):
    sentences = re.split(r'[.\n]', text)
    safety_keywords = get_safety_keywords()
    warnings = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_lower = sentence.lower()
        
        """# Error for some reason
        # Check for safety keywords from the databases
        if any(keyword in sentence_lower for keyword in safety_keywords):
            warnings.append(sentence)
            continue"""

        flat_keywords = set()
        for phrase in safety_keywords:
            flat_keywords.update(phrase.split())

        sentence_words = re.findall(r'\w+', sentence_lower)
        if any(word in sentence_words for word in flat_keywords):
            warnings.append(sentence)


            
        """# Check for standard warning indicators
        if any(indicator in sentence_lower for indicator in ['safety', 'warning', 'caution', 'danger', 
                                                    'hazard', 'risk', 'protect', 'secure']):
            warnings.append(sentence)"""
    
    return list(set(warnings))

    

# Example usage:
if __name__ == "__main__":
    text = extract_text_from_pdf("metal_jet_s100_English.pdf")
    print("\n=== PARAMETERS ===")
    for param in extract_parameters(text):
        print("-", param)
    """print("\n=== SAFETY WARNINGS ===")
    for warn in extract_safety(text):
        print("-", warn)"""
