# Technical Parameter & Safety Extraction with NER

This project provides tools for extracting technical parameters and safety-related information from technical documents (PDFs, CSVs) using a combination of pattern matching, classical NLP, and deep learning (TensorFlow, BERT). It also supports the generation and use of synthetic data for model training and evaluation.

---

## File Descriptions

### 1. `param&safety.py`
- **Purpose:**  
  Extracts technical parameters (with units) and safety-related sentences from PDF documents using pattern matching and keyword mining.
- **Key Features:**
  - Extracts text from PDFs.
  - Loads unit patterns from ECLASS CSV files.
  - Extracts parameters (e.g., temperature, pressure) using regex.
  - Extracts safety warnings/indicators using keyword mining from ECLASS CSVs.

---

### 2. `Parameters_BERT.py`
- **Purpose:**  
  Uses a BERT-based NER model to extract parameter entities from PDF text, leveraging pre-trained or fine-tuned transformer models.
- **Key Features:**
  - Extracts text from PDFs.
  - Loads unit patterns from ECLASS CSV.
  - Runs a BERT NER pipeline to identify parameter entities.
  - Filters NER results by unit patterns.

---

### 3. `Parameters_Tensorflow.py`
- **Purpose:**  
  Implements a TensorFlow-based deep learning model to classify and extract parameter sentences from technical documents.
- **Key Features:**
  - Extracts text from PDFs.
  - Loads unit patterns from ECLASS CSV.
  - Preprocesses text and tokenizes sentences.
  - Trains a simple CNN-based classifier to identify parameter sentences.
  - Extracts parameters from classified sentences using regex.

---

### 4. `Parameters_Tensorflow_SyntheticData.py`
- **Purpose:**  
  Similar to `Parameters_Tensorflow.py`, but uses synthetic data for model training, enabling robust parameter extraction even with limited real data.
- **Key Features:**
  - Imports and uses synthetic parameter sentences for training.
  - Trains and applies the TensorFlow model as above.

---

### 5. `Synthetic_data.py`
- **Purpose:**  
  Generates synthetic sentences containing technical parameters and units, based on ECLASS CSV data.
- **Key Features:**
  - Loads parameter names and units from ECLASS CSV.
  - Randomly generates sentences using various templates.
  - Used for data augmentation and model training.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install dependencies:**
   - If you have a `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```
   - Or, install manually:
     ```bash
     pip install pdfplumber pandas tensorflow numpy transformers
     ```

3. **Ensure ECLASS CSV files are present in the `Datasets/1730150000enUSbasicCSV01/` directory.**

---

## Usage

- **Extract parameters and safety info (pattern-based):**
  ```bash
  python param&safety.py
  ```

- **Extract parameters using BERT NER:**
  ```bash
  python Parameters_BERT.py
  ```

- **Train and extract parameters using TensorFlow (real data):**
  ```bash
  python Parameters_Tensorflow.py
  ```

- **Train and extract parameters using TensorFlow (synthetic data):**
  ```bash
  python Parameters_Tensorflow_SyntheticData.py
  ```

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---
