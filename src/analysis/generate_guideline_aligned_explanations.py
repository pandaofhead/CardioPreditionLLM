#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Guideline-Aligned Explanation Generator using LLM API.
This script generates medical guideline-aligned explanations by combining model predictions
with standard medical guidelines (ACC/AHA) for cardiovascular disease risk assessment.
"""

import os
import json
import time
from pathlib import Path
import openai
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]

# Medical guidelines reference
MEDICAL_GUIDELINES = {
    "blood_pressure": {
        "normal": "< 120/80 mm Hg",
        "elevated": "120-129/< 80 mm Hg",
        "stage1": "130-139/80-89 mm Hg",
        "stage2": "≥ 140/90 mm Hg",
        "crisis": "> 180/120 mm Hg"
    },
    "cholesterol": {
        "total": {
            "normal": "< 200 mg/dL",
            "borderline": "200-239 mg/dL",
            "high": "≥ 240 mg/dL"
        }
    },
    "heart_rate": {
        "resting": {
            "normal": "60-100 bpm",
            "bradycardia": "< 60 bpm",
            "tachycardia": "> 100 bpm"
        },
        "max": "220 - age (theoretical maximum)"
    },
    "fasting_glucose": {
        "normal": "< 100 mg/dL",
        "prediabetes": "100-125 mg/dL",
        "diabetes": "≥ 126 mg/dL"
    }
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
                    {"role": "system", "content": "You are a medical AI assistant helping to interpret cardiovascular disease risk predictions. Your explanations should align with established medical guidelines and standard clinical practice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Balance between creativity and consistency
                max_tokens=1500,  # Allow for detailed responses
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

def classify_measurements(clinical_features, demographics):
    """Classify patient measurements according to medical guidelines"""
    classifications = {}
    
    # Blood pressure classification
    bp = clinical_features['resting_blood_pressure']
    if bp < 120:
        classifications['blood_pressure'] = 'normal'
    elif 120 <= bp < 130:
        classifications['blood_pressure'] = 'elevated'
    elif 130 <= bp < 140:
        classifications['blood_pressure'] = 'stage1'
    elif 140 <= bp < 180:
        classifications['blood_pressure'] = 'stage2'
    else:
        classifications['blood_pressure'] = 'crisis'
    
    # Cholesterol classification
    chol = clinical_features['serum_cholesterol']
    if chol < 200:
        classifications['cholesterol'] = 'normal'
    elif 200 <= chol < 240:
        classifications['cholesterol'] = 'borderline'
    else:
        classifications['cholesterol'] = 'high'
    
    # Heart rate classification
    hr = clinical_features['max_heart_rate']
    theoretical_max = 220 - demographics['age']
    classifications['heart_rate'] = {
        'value': hr,
        'theoretical_max': theoretical_max,
        'percentage': round((hr / theoretical_max) * 100, 1)
    }
    
    return classifications

def generate_guideline_prompt(patient_desc, classifications):
    """Generate a prompt for guideline-aligned analysis"""
    # Extract patient information
    patient_id = patient_desc['patient_id']
    demographics = patient_desc['demographics']
    clinical_features = patient_desc['clinical_features']
    predictions = patient_desc['predictions']
    
    # Create the prompt
    prompt = f"""You are a medical AI assistant helping to interpret cardiovascular disease (CVD) risk predictions in accordance with established medical guidelines.

Patient Information:
- ID: {patient_id}
- Age: {demographics['age']} years
- Sex: {demographics['sex']}

Clinical Features and Guideline-Based Classifications:

1. Blood Pressure: {clinical_features['resting_blood_pressure']} mm Hg
   - Classification: {classifications['blood_pressure'].upper()}
   - Guideline Reference: {MEDICAL_GUIDELINES['blood_pressure'][classifications['blood_pressure']]}

2. Cholesterol: {clinical_features['serum_cholesterol']} mg/dL
   - Classification: {classifications['cholesterol'].upper()}
   - Guideline Reference: {MEDICAL_GUIDELINES['cholesterol']['total'][classifications['cholesterol']]}

3. Heart Rate:
   - Maximum: {clinical_features['max_heart_rate']} bpm
   - Theoretical Maximum (220 - age): {classifications['heart_rate']['theoretical_max']} bpm
   - Percentage of Max: {classifications['heart_rate']['percentage']}%

Other Clinical Features:
- Chest Pain Type: {clinical_features['chest_pain_type']}
- Fasting Blood Sugar: {clinical_features['fasting_blood_sugar']}
- Resting ECG: {clinical_features['resting_ecg']}
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

Please provide a comprehensive analysis that aligns with current medical guidelines:

1. Risk Assessment:
   - Evaluate the patient's CVD risk factors according to ACC/AHA guidelines
   - Compare the model's predictions with standard risk assessment criteria
   - Identify any discrepancies between model predictions and guideline-based assessment

2. Treatment Recommendations:
   - Provide guideline-based recommendations for each risk factor
   - Suggest appropriate lifestyle modifications and potential interventions
   - Include specific target values based on current guidelines

3. Monitoring Plan:
   - Recommend appropriate follow-up intervals
   - Specify which parameters should be monitored
   - Suggest when to consider additional testing or specialist referral

4. Special Considerations:
   - Discuss any age, gender, or condition-specific recommendations
   - Address potential contraindications or precautions
   - Consider comorbidity management if relevant

Your response should integrate both the model's predictions and standard medical guidelines to provide evidence-based recommendations that a healthcare provider can use in clinical practice.
"""
    
    return prompt

def process_patient_descriptions(input_dir, output_dir):
    """Process all patient descriptions and generate guideline-aligned explanations"""
    logging.info("Starting patient description processing")
    
    # Load API key
    try:
        api_key = load_env()
        logging.info("API key loaded successfully")
    except Exception as e:
        logging.error(f"Error loading API key: {str(e)}")
        return
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created: {output_dir}")
    except Exception as e:
        logging.error(f"Error creating output directory: {str(e)}")
        return
    
    # Get all patient description files
    try:
        description_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_description.json')])
        logging.info(f"Found {len(description_files)} patient descriptions to process")
    except Exception as e:
        logging.error(f"Error listing input directory: {str(e)}")
        return
    
    # Process each patient description
    for desc_file in tqdm(description_files, desc="Generating guideline-aligned explanations"):
        patient_id = desc_file.split('_')[1]  # Extract patient ID from filename
        logging.info(f"Processing patient {patient_id}")
        
        try:
            # Read patient description
            with open(os.path.join(input_dir, desc_file), 'r') as f:
                patient_desc = json.load(f)
            logging.info(f"Loaded description for patient {patient_id}")
            
            # Classify measurements according to guidelines
            classifications = classify_measurements(patient_desc['clinical_features'], patient_desc['demographics'])
            logging.info(f"Classifications generated for patient {patient_id}")
            
            # Generate guideline-aligned prompt
            prompt = generate_guideline_prompt(patient_desc, classifications)
            logging.info(f"Prompt generated for patient {patient_id}")
            
            # Get LLM interpretation
            explanation = get_llm_response(api_key, prompt)
            logging.info(f"LLM explanation received for patient {patient_id}")
            
            # Save explanation
            output_file = f"patient_{patient_id}_guideline_aligned.txt"
            with open(os.path.join(output_dir, output_file), 'w') as f:
                f.write(explanation)
            
            # Also save as part of a structured JSON
            result = {
                "patient_id": f"P{patient_id}",
                "classifications": classifications,
                "prompt": prompt,
                "guideline_aligned_explanation": explanation
            }
            
            json_output_file = f"patient_{patient_id}_guideline_aligned.json"
            with open(os.path.join(output_dir, json_output_file), 'w') as f:
                json.dump(result, f, indent=2)
            
            logging.info(f"Results saved for patient {patient_id}")
            
            # Add a small delay between requests to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error processing patient {patient_id}: {str(e)}")
            continue

def main():
    """Main function to generate guideline-aligned explanations"""
    logging.info("Starting guideline-aligned explanation generation...")
    
    # Set up directories
    input_dir = os.path.join(project_root, 'models', 'llm', 'patient_descriptions')
    output_dir = os.path.join(project_root, 'models', 'llm', 'guideline_aligned_explanations')
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Process patient descriptions
    process_patient_descriptions(input_dir, output_dir)
    
    logging.info(f"Guideline-aligned explanations generated and saved in: {output_dir}")

if __name__ == "__main__":
    main() 