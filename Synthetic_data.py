"""
Synthetic_data.py

Generates synthetic sentences containing technical parameters and units, based on ECLASS CSV data.

Features:
- Loads parameter names and units from ECLASS CSV.
- Randomly generates sentences using various templates.
- Used for data augmentation and model training.
"""

import pandas as pd
import random

# Function to generate and print synthetic parameter sentences
def get_synthetic_parameter_sentences(num_sentences=30):
    # Load the ECLASS units CSV file
    data = pd.read_csv("Datasets/1730150000enUSbasicCSV01/ECLASS15_0_UN_en.csv", delimiter=';')

    # Get a list of parameter names (quantities)
    parameter_names = data['NameOfDedicatedQuantity'].dropna().unique().tolist()

    # Get a list of units (from both ShortName and SINotation columns)
    unit_names = data['ShortName'].dropna().unique().tolist()
    unit_names += data['SINotation'].dropna().unique().tolist()
    # Remove any empty or non-string units
    unit_names = [u for u in unit_names if isinstance(u, str) and u.strip()]

    # Define sentence templates for variety
    sentence_templates = [
        "The {param} is {value} {unit}.",
        "{param} measured: {value}{unit}",
        "Set {param} to {value} {unit}.",
        "Typical {param} ranges from {value1} to {value2} {unit}.",
        "Maximum {param}: {value}{unit}",
        "Minimum {param}: {value}{unit}",
        "Average {param} is about {value} {unit}.",
        "The device operates at {value} {unit} {param}.",
        "This is a fake sentence.",
    ]

    # Helper function to generate a random value (float or int)
    def random_value():
        if random.random() < 0.5:
            return str(round(random.uniform(0.01, 1000), 2))  # float value
        else:
            return str(random.randint(1, 1000))  # integer value

    # Generate and print the synthetic sentences
    for _ in range(num_sentences):
        param = random.choice(parameter_names)
        unit = random.choice(unit_names)
        template = random.choice(sentence_templates)
        value1 = random_value()
        value2 = random_value()
        # For range templates, make sure value1 < value2
        if float(value1) > float(value2):
            value1, value2 = value2, value1
        # Fill in the template with the chosen values
        sentence = template.format(param=param, unit=unit, value=value1, value1=value1, value2=value2)
        sentences = []
        sentences.append(sentence)
        #print(sentence)
    return sentences

"""if __name__ == "__main__":
    print("\n=== 30 SYNTHETIC PARAMETER SENTENCES ===")
    get_synthetic_parameter_sentences(30)"""