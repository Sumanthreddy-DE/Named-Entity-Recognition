"""
Parameters_Tensorflow_SyntheticData.py

Similar to Parameters_Tensorflow.py, but uses synthetic data for model training, enabling robust parameter extraction even with limited real data.

Features:
- Imports and uses synthetic parameter sentences for training.
- Trains and applies the TensorFlow model as above.
"""

import pdfplumber
import re
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
from Synthetic_data import get_synthetic_parameter_sentences

"""tf.random.set_seed(42)
np.random.seed(42) #data shuffling during model training
random.seed(42)"""


"""run_information_extraction()
├── extract_text_from_pdf()
├── train_model()
│   ├── create_model()
│   └── Preprocess training data
└── extract_parameters_with_tensorflow()
    ├── ML-based detection
    └── Pattern-based fallback"""


# 1. Extract all text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:10]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_unit_patterns():
    patterns = []
    unit_info = {}
    with open('Datasets/1730150000enUSbasicCSV01/ECLASS15_0_UN_en.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            short_name = row.get('ShortName', '').strip()
            si_notation = row.get('SINotation', '').strip()
            
            if short_name:
                escaped_name = re.escape(short_name)
                pattern = rf"(\d+(?:\.\d+)?\s*{escaped_name})"
                patterns.append(pattern)
                unit_info[pattern] = {'unit': short_name}
            
            if si_notation:
                escaped_notation = re.escape(si_notation)
                pattern = rf"(\d+(?:\.\d+)?\s*{escaped_notation})"
                patterns.append(pattern)
                unit_info[pattern] = {'unit': si_notation}
    
    return list(set(patterns)), unit_info


def preprocess_text(text, tokenizer, max_length=100):
    sentences = re.split(r'[.\n]', text)
    processed_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        sequence = tokenizer.texts_to_sequences([sentence])[0]
        padded = pad_sequences([sequence], maxlen=max_length, padding='post')
        processed_sentences.append({
            'sequence': padded[0],
            'sentence': sentence
        })
    
    return processed_sentences

"""[
    {
        'sequence': [1, 2, 3, 4, 0, 0],
        'sentence': 'The temperature is 25°C'
    },
    {
        'sequence': [1, 5, 3, 6, 7, 0],
        'sentence': 'The pressure is 2.5 bar'
    }
]
"""



def create_model(text_data, labels, vocab_size, max_length):
    inputs = layers.Input(shape=(max_length,))
    embedding = layers.Embedding(vocab_size, 64)(inputs)
    conv1 = layers.Conv1D(128, 3, activation='relu')(embedding)
    pool1 = layers.GlobalMaxPooling1D()(conv1)
    dense1 = layers.Dense(64, activation='relu')(pool1) #non linear
    outputs = layers.Dense(1, activation='sigmoid')(dense1) #binary output
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    labels = np.array(labels)
    model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    return model, tokenizer



def extract_parameters_with_tensorflow(text, model, tokenizer, max_length=100):
    processed_sentences = preprocess_text(text, tokenizer, max_length)
    parameters = []
    
    # Get unit patterns
    patterns, unit_info = get_unit_patterns()
    
    for processed in processed_sentences:
        prediction = model.predict(tf.expand_dims(processed['sequence'], 0))
        
        if prediction[0][0] > 0.5:
            sentence = processed['sentence']
            
            # Try to find parameters with units
            for pattern in patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    # Clean up the match and ensure proper formatting
                    match = match.strip()
                    if match not in parameters:
                        parameters.append(match)
            
            """# for numbers without units
            number_matches = re.findall(r'\d+(?:\.\d+)?', sentence)
            for num in number_matches:
                if not any(num in param for param in parameters):
                    parameters.append(num)"""
    
    return sorted(list(set(parameters)), key=lambda x: float(re.search(r'\d+(?:\.\d+)?', x).group()))



# Example usage:
if __name__ == "__main__":
    # Use synthetic data for training
    text_data = get_synthetic_parameter_sentences(100)
    labels = [1] * len(text_data)  # All synthetic sentences are parameter sentences
    print("Sample synthetic training data:")
    for s in text_data[:5]:
        print(s)
    vocab_size = 2000
    max_length = 100
    model, tokenizer = create_model(text_data, labels, vocab_size, max_length)
    
    print("\n=== EXTRACTED PARAMETERS ===")
    parameters = extract_parameters_with_tensorflow(text, model, tokenizer)
    for param in parameters:
        print(f"- {param}")