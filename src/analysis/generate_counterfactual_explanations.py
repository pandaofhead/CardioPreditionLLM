#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Counterfactual Explanation Generator using LLM API.
This script generates "if...then..." type explanations for patients based on their key modifiable features.
"""

import os
import json
import time
from pathlib import Path
import openai
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]

# Feature descriptions and units
FEATURE_DESCRIPTIONS = {
    'age': {'name': 'Age', 'unit': 'years', 'modifiable': False},
    'sex': {'name': 'Sex', 'unit': 'male/female', 'modifiable': False},
    'cp': {'name': 'Chest Pain Type', 'unit': '0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic', 'modifiable': False},
    'trestbps': {'name': 'Resting Blood Pressure', 'unit': 'mm Hg', 'modifiable': True},
    'chol': {'name': 'Serum Cholesterol', 'unit': 'mg/dl', 'modifiable': True},
    'fbs': {'name': 'Fasting Blood Sugar', 'unit': '>120 mg/dl (1 = true; 0 = false)', 'modifiable': True},
    'restecg': {'name': 'Resting ECG', 'unit': '0: normal, 1: ST-T wave abnormality, 2: probable/definite left ventricular hypertrophy', 'modifiable': False},
    'thalach': {'name': 'Maximum Heart Rate', 'unit': 'bpm', 'modifiable': True},
    'exang': {'name': 'Exercise-Induced Angina', 'unit': '1 = yes; 0 = no', 'modifiable': True},
    'oldpeak': {'name': 'ST Depression', 'unit': 'mm', 'modifiable': True},
    'slope': {'name': 'ST Slope', 'unit': '0: upsloping, 1: flat, 2: downsloping', 'modifiable': False},
    'ca': {'name': 'Number of Major Vessels', 'unit': '0-3', 'modifiable': False},
    'thal': {'name': 'Thalassemia', 'unit': '1: normal, 2: fixed defect, 3: reversible defect', 'modifiable': False}
}

# Chest pain type descriptions
CP_DESCRIPTIONS = {
    0: 'typical angina',
    1: 'atypical angina',
    2: 'non-anginal pain',
    3: 'asymptomatic'
}

# Resting ECG descriptions
RESTECG_DESCRIPTIONS = {
    0: 'normal',
    1: 'having ST-T wave abnormality',
    2: 'showing probable or definite left ventricular hypertrophy'
}

# Slope descriptions
SLOPE_DESCRIPTIONS = {
    0: 'upsloping',
    1: 'flat',
    2: 'downsloping'
}

# Thalassemia descriptions
THAL_DESCRIPTIONS = {
    1: 'normal',
    2: 'fixed defect',
    3: 'reversible defect'
}

def load_env():
    """Load environment variables from .env.local file"""
    env_path = os.path.join(project_root, '.env.local')
    load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return api_key

def get_llm_response(api_key, prompt, max_retries=3, delay=2):
    """Get response from LLM API with retry mechanism"""
    openai.api_key = api_key
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Using GPT-4 for high-quality medical interpretations
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant helping to interpret cardiovascular disease risk predictions. Provide clear, accurate, and actionable medical insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Balance between creativity and consistency
                max_tokens=1000,  # Allow for detailed responses
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff

def identify_modifiable_features(patient_desc, shap_values):
    """Identify key modifiable features for counterfactual analysis"""
    # Get all modifiable features
    modifiable_features = [feature for feature, info in FEATURE_DESCRIPTIONS.items() if info['modifiable']]
    
    # Get SHAP values for modifiable features
    feature_shap = {feature: shap_values.get(feature, 0) for feature in modifiable_features}
    
    # Sort by absolute SHAP value to find most impactful features
    sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Return top 3 most impactful modifiable features
    return sorted_features[:3]

def generate_counterfactual_prompt(patient_desc, key_features):
    """Generate a prompt for counterfactual analysis"""
    # Extract patient information
    patient_id = patient_desc['patient_id']
    demographics = patient_desc['demographics']
    clinical_features = patient_desc['clinical_features']
    predictions = patient_desc['predictions']
    
    # Format key features for the prompt
    feature_descriptions = []
    for feature, shap_value in key_features:
        feature_info = FEATURE_DESCRIPTIONS[feature]
        current_value = None
        
        # Get current value based on feature name
        if feature == 'trestbps':
            current_value = clinical_features['resting_blood_pressure']
        elif feature == 'chol':
            current_value = clinical_features['serum_cholesterol']
        elif feature == 'fbs':
            current_value = "High (>120 mg/dl)" if clinical_features['fasting_blood_sugar'] == "High (>120 mg/dl)" else "Normal (≤120 mg/dl)"
        elif feature == 'thalach':
            current_value = clinical_features['max_heart_rate']
        elif feature == 'exang':
            current_value = "Yes" if clinical_features['exercise_induced_angina'] == "Yes" else "No"
        elif feature == 'oldpeak':
            current_value = clinical_features['st_depression']
        
        # Add feature description with current value
        feature_descriptions.append(f"{feature_info['name']}: {current_value} {feature_info['unit']} (SHAP value: {shap_value:.4f})")
    
    # Create the prompt
    prompt = f"""You are a medical AI assistant helping to interpret cardiovascular disease (CVD) risk predictions.

Patient Information:
- ID: {patient_id}
- Age: {demographics['age']} years
- Sex: {demographics['sex']}

Clinical Features:
- Chest Pain Type: {clinical_features['chest_pain_type']}
- Resting Blood Pressure: {clinical_features['resting_blood_pressure']} mm Hg
- Serum Cholesterol: {clinical_features['serum_cholesterol']} mg/dl
- Fasting Blood Sugar: {clinical_features['fasting_blood_sugar']}
- Resting ECG: {clinical_features['resting_ecg']}
- Maximum Heart Rate: {clinical_features['max_heart_rate']} bpm
- Exercise-Induced Angina: {clinical_features['exercise_induced_angina']}
- ST Depression: {clinical_features['st_depression']}
- ST Slope: {clinical_features['st_slope']}
- Number of Major Vessels: {clinical_features['number_of_vessels']}
- Thalassemia: {clinical_features['thalassemia']}

Model Predictions:
1. Random Forest Model:
   - Prediction: {predictions['random_forest']['prediction']}
   - Confidence: {predictions['random_forest']['confidence']:.2f}

2. XGBoost Model:
   - Prediction: {predictions['xgboost']['prediction']}
   - Confidence: {predictions['xgboost']['confidence']:.2f}

Key Modifiable Features for Counterfactual Analysis:
{chr(10).join(feature_descriptions)}

Please provide counterfactual explanations for this patient by answering the following questions:

1. For each of the key modifiable features identified above, explain how changing the value might affect the patient's CVD risk. Use the format "If [feature] is changed from [current value] to [suggested value], then [explanation of potential impact on risk]."

2. Provide specific, actionable recommendations for each modifiable feature, including:
   - Target values or ranges to aim for
   - Lifestyle changes or interventions that could help achieve these targets
   - Expected timeline for seeing improvements
   - Potential challenges and how to overcome them

3. Estimate the potential impact on risk score if all recommended changes are implemented.

4. Discuss any interactions between the different modifiable features and how addressing them together might have a greater impact than addressing them individually.

Your response should be in clear, non-technical language that a healthcare provider could use to explain the situation to a patient.
"""
    
    return prompt

def process_patient_descriptions(input_dir, output_dir):
    """Process all patient descriptions and generate counterfactual explanations"""
    # Load API key
    api_key = load_env()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all patient description files
    description_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_description.json')])
    
    print(f"Found {len(description_files)} patient descriptions to process")
    
    # Process each patient description
    for desc_file in tqdm(description_files, desc="Generating counterfactual explanations"):
        patient_id = desc_file.split('_')[1]  # Extract patient ID from filename
        
        # Read patient description
        with open(os.path.join(input_dir, desc_file), 'r') as f:
            patient_desc = json.load(f)
        
        try:
            # Identify key modifiable features using SHAP values from both models
            rf_shap_values = patient_desc['predictions']['random_forest']['shap_values']
            xgb_shap_values = patient_desc['predictions']['xgboost']['shap_values']
            
            # Combine SHAP values from both models (average)
            combined_shap = {}
            for feature in rf_shap_values:
                combined_shap[feature] = (rf_shap_values[feature] + xgb_shap_values[feature]) / 2
            
            # Identify key modifiable features
            key_features = identify_modifiable_features(patient_desc, combined_shap)
            
            # Generate counterfactual prompt
            prompt = generate_counterfactual_prompt(patient_desc, key_features)
            
            # Get LLM interpretation
            counterfactual = get_llm_response(api_key, prompt)
            
            # Save counterfactual explanation
            output_file = f"patient_{patient_id}_counterfactual.txt"
            with open(os.path.join(output_dir, output_file), 'w') as f:
                f.write(counterfactual)
            
            # Also save as part of a structured JSON
            result = {
                "patient_id": f"P{patient_id}",
                "key_modifiable_features": [{"feature": feature, "shap_value": shap_value} for feature, shap_value in key_features],
                "prompt": prompt,
                "counterfactual_explanation": counterfactual
            }
            
            json_output_file = f"patient_{patient_id}_counterfactual.json"
            with open(os.path.join(output_dir, json_output_file), 'w') as f:
                json.dump(result, f, indent=2)
            
            # Add a small delay between requests to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
            continue

def main():
    """Main function to generate counterfactual explanations"""
    print("Starting counterfactual explanation generation...")
    
    # Set up directories
    input_dir = os.path.join(project_root, 'models', 'llm', 'patient_descriptions')
    output_dir = os.path.join(project_root, 'models', 'llm', 'counterfactual_explanations')
    
    # Process patient descriptions
    process_patient_descriptions(input_dir, output_dir)
    
    print(f"\nCounterfactual explanations generated and saved in: {output_dir}")

if __name__ == "__main__":
    main() 