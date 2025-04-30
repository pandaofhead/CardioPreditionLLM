#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patient Description Generator for LLM Interpretation.
This script creates structured patient descriptions with prediction results, 
confidence scores, and SHAP values for LLM interpretation.
"""

import os
import pandas as pd
import numpy as np
import joblib
import shap
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]

# Feature descriptions for human-readable output
FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
    'trestbps': 'Resting blood pressure in mm Hg on admission to the hospital',
    'chol': 'Serum cholesterol in mg/dl',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)'
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

def load_models():
    """Load trained models and their SHAP values"""
    # Load models
    rf_pipeline = joblib.load(os.path.join(project_root, 'models', 'traditional_ml', 'random_forest.pkl'))
    xgb_pipeline = joblib.load(os.path.join(project_root, 'models', 'traditional_ml', 'xgboost.pkl'))
    
    # Extract models from pipelines
    rf_model = rf_pipeline.named_steps['classifier']
    xgb_model = xgb_pipeline.named_steps['classifier']
    
    # Load feature importance
    feature_importance_dir = os.path.join(project_root, 'models', 'traditional_ml', 'feature_importance')
    rf_importance = pd.read_csv(os.path.join(feature_importance_dir, 'random_forest_feature_importance.csv'))
    xgb_importance = pd.read_csv(os.path.join(feature_importance_dir, 'xgboost_feature_importance.csv'))
    
    return rf_model, xgb_model, rf_pipeline, xgb_pipeline, rf_importance, xgb_importance

def load_validation_data():
    """Load validation dataset for SHAP analysis"""
    data_dir = os.path.join(project_root, 'data', 'processed')
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))
    return X_val, y_val

def get_shap_values(model, pipeline, X):
    """Get SHAP values for a model"""
    # Transform the data using the pipeline's preprocessing steps
    X_transformed = pipeline.named_steps['scaler'].transform(X)
    X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    # For binary classification, we want the positive class SHAP values
    if isinstance(shap_values, list):
        # For Random Forest, take the positive class values
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        # For Random Forest, take the positive class values
        shap_values = shap_values[:, :, 1]
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"First SHAP value: {shap_values[0][0]}")
    
    return shap_values, X_transformed

def get_prediction_confidence(model, X_transformed):
    """Get prediction and confidence score for a model"""
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_transformed)
        prediction = model.predict(X_transformed)
        confidence = np.max(proba, axis=1)
    else:
        # For models without predict_proba, use decision function
        decision = model.decision_function(X_transformed)
        prediction = (decision > 0).astype(int)
        confidence = np.abs(decision) / (np.abs(decision).max() + 1e-10)
    
    return prediction, confidence

def create_patient_description(patient_data, rf_model, xgb_model, rf_pipeline, xgb_pipeline, 
                              rf_importance, xgb_importance, patient_idx=0):
    """Create a structured patient description with prediction results and SHAP values"""
    # Get patient data
    patient = patient_data.iloc[patient_idx:patient_idx+1]
    
    # Get SHAP values
    print("\nGetting Random Forest SHAP values...")
    rf_shap_values, rf_X_transformed = get_shap_values(rf_model, rf_pipeline, patient)
    print("\nGetting XGBoost SHAP values...")
    xgb_shap_values, xgb_X_transformed = get_shap_values(xgb_model, xgb_pipeline, patient)
    
    # Get predictions and confidence
    rf_pred, rf_conf = get_prediction_confidence(rf_model, rf_X_transformed)
    xgb_pred, xgb_conf = get_prediction_confidence(xgb_model, xgb_X_transformed)
    
    # Create feature importance dictionaries
    rf_feature_imp = dict(zip(rf_importance['feature'], rf_importance['importance']))
    xgb_feature_imp = dict(zip(xgb_importance['feature'], xgb_importance['importance']))
    
    # Create SHAP value dictionaries
    rf_shap_dict = {}
    xgb_shap_dict = {}
    
    for i, col in enumerate(patient.columns):
        # Convert numpy array values to Python float
        rf_val = rf_shap_values[i] if len(rf_shap_values.shape) == 1 else rf_shap_values[0][i]
        xgb_val = xgb_shap_values[i] if len(xgb_shap_values.shape) == 1 else xgb_shap_values[0][i]
        print(f"\nFeature: {col}")
        print(f"RF SHAP value: {rf_val}")
        print(f"XGB SHAP value: {xgb_val}")
        rf_shap_dict[col] = float(rf_val)
        xgb_shap_dict[col] = float(xgb_val)
    
    # Create patient description
    patient_desc = {
        "patient_id": f"P{patient_idx:04d}",
        "demographics": {
            "age": int(patient['age'].values[0]),
            "sex": "Male" if patient['sex'].values[0] == 1 else "Female"
        },
        "clinical_features": {
            "chest_pain_type": CP_DESCRIPTIONS[int(patient['cp'].values[0])],
            "resting_blood_pressure": int(patient['trestbps'].values[0]),
            "serum_cholesterol": int(patient['chol'].values[0]),
            "fasting_blood_sugar": "High (>120 mg/dl)" if patient['fbs'].values[0] == 1 else "Normal (≤120 mg/dl)",
            "resting_ecg": RESTECG_DESCRIPTIONS[int(patient['restecg'].values[0])],
            "max_heart_rate": int(patient['thalach'].values[0]),
            "exercise_induced_angina": "Yes" if patient['exang'].values[0] == 1 else "No",
            "st_depression": float(patient['oldpeak'].values[0]),
            "st_slope": SLOPE_DESCRIPTIONS[int(patient['slope'].values[0])],
            "number_of_vessels": int(patient['ca'].values[0]),
            "thalassemia": THAL_DESCRIPTIONS[int(patient['thal'].values[0])]
        },
        "predictions": {
            "random_forest": {
                "prediction": "High CVD Risk" if rf_pred[0] == 1 else "Low CVD Risk",
                "confidence": float(rf_conf[0]),
                "feature_importance": rf_feature_imp,
                "shap_values": rf_shap_dict
            },
            "xgboost": {
                "prediction": "High CVD Risk" if xgb_pred[0] == 1 else "Low CVD Risk",
                "confidence": float(xgb_conf[0]),
                "feature_importance": xgb_feature_imp,
                "shap_values": xgb_shap_dict
            }
        }
    }
    
    return patient_desc

def generate_llm_prompt(patient_desc):
    """Generate a prompt for LLM based on patient description"""
    # Get top contributing and risk-reducing factors
    rf_top_contributing = sorted(
        [(k, v) for k, v in patient_desc['predictions']['random_forest']['shap_values'].items() if v > 0],
        key=lambda x: x[1], reverse=True
    )[:3]
    
    rf_top_reducing = sorted(
        [(k, v) for k, v in patient_desc['predictions']['random_forest']['shap_values'].items() if v < 0],
        key=lambda x: x[1]
    )[:3]
    
    xgb_top_contributing = sorted(
        [(k, v) for k, v in patient_desc['predictions']['xgboost']['shap_values'].items() if v > 0],
        key=lambda x: x[1], reverse=True
    )[:3]
    
    xgb_top_reducing = sorted(
        [(k, v) for k, v in patient_desc['predictions']['xgboost']['shap_values'].items() if v < 0],
        key=lambda x: x[1]
    )[:3]
    
    # Format feature names for display
    def format_feature_name(feature):
        if feature == 'cp':
            return 'Chest Pain Type'
        elif feature == 'trestbps':
            return 'Resting Blood Pressure'
        elif feature == 'chol':
            return 'Serum Cholesterol'
        elif feature == 'fbs':
            return 'Fasting Blood Sugar'
        elif feature == 'restecg':
            return 'Resting ECG'
        elif feature == 'thalach':
            return 'Maximum Heart Rate'
        elif feature == 'exang':
            return 'Exercise-Induced Angina'
        elif feature == 'oldpeak':
            return 'ST Depression'
        elif feature == 'slope':
            return 'ST Slope'
        elif feature == 'ca':
            return 'Number of Major Vessels'
        elif feature == 'thal':
            return 'Thalassemia'
        elif feature == 'age':
            return 'Age'
        elif feature == 'sex':
            return 'Sex'
        else:
            return feature
    
    prompt = f"""You are a medical AI assistant helping to interpret cardiovascular disease (CVD) risk predictions.

Patient Information:
- ID: {patient_desc['patient_id']}
- Age: {patient_desc['demographics']['age']} years
- Sex: {patient_desc['demographics']['sex']}

Clinical Features:
- Chest Pain Type: {patient_desc['clinical_features']['chest_pain_type']}
- Resting Blood Pressure: {patient_desc['clinical_features']['resting_blood_pressure']} mm Hg
- Serum Cholesterol: {patient_desc['clinical_features']['serum_cholesterol']} mg/dl
- Fasting Blood Sugar: {patient_desc['clinical_features']['fasting_blood_sugar']}
- Resting ECG: {patient_desc['clinical_features']['resting_ecg']}
- Maximum Heart Rate: {patient_desc['clinical_features']['max_heart_rate']} bpm
- Exercise-Induced Angina: {patient_desc['clinical_features']['exercise_induced_angina']}
- ST Depression: {patient_desc['clinical_features']['st_depression']}
- ST Slope: {patient_desc['clinical_features']['st_slope']}
- Number of Major Vessels: {patient_desc['clinical_features']['number_of_vessels']}
- Thalassemia: {patient_desc['clinical_features']['thalassemia']}

Model Predictions:
1. Random Forest Model:
   - Prediction: {patient_desc['predictions']['random_forest']['prediction']}
   - Confidence: {patient_desc['predictions']['random_forest']['confidence']:.2f}
   - Top Contributing Factors (positive SHAP values):
     {', '.join([f"{format_feature_name(k)} ({v:.4f})" for k, v in rf_top_contributing])}
   - Top Risk-Reducing Factors (negative SHAP values):
     {', '.join([f"{format_feature_name(k)} ({v:.4f})" for k, v in rf_top_reducing])}

2. XGBoost Model:
   - Prediction: {patient_desc['predictions']['xgboost']['prediction']}
   - Confidence: {patient_desc['predictions']['xgboost']['confidence']:.2f}
   - Top Contributing Factors (positive SHAP values):
     {', '.join([f"{format_feature_name(k)} ({v:.4f})" for k, v in xgb_top_contributing])}
   - Top Risk-Reducing Factors (negative SHAP values):
     {', '.join([f"{format_feature_name(k)} ({v:.4f})" for k, v in xgb_top_reducing])}

Please provide:
1. A clinical interpretation of the patient's CVD risk based on the model predictions
2. An explanation of the key factors contributing to the risk assessment
3. Specific recommendations for risk management based on the patient's profile
4. A discussion of any uncertainties or limitations in the prediction
5. Suggestions for additional tests or information that might improve the assessment

Your response should be in clear, non-technical language that a healthcare provider could use to explain the situation to a patient.
"""
    
    return prompt

def main():
    """Main function to generate patient descriptions and LLM prompts"""
    print("Loading models and data...")
    rf_model, xgb_model, rf_pipeline, xgb_pipeline, rf_importance, xgb_importance = load_models()
    X_val, y_val = load_validation_data()
    
    # Create output directory
    output_dir = os.path.join(project_root, 'models', 'llm', 'patient_descriptions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate descriptions for a few patients
    num_patients = min(5, len(X_val))
    for i in range(num_patients):
        print(f"Generating description for patient {i}...")
        patient_desc = create_patient_description(X_val, rf_model, xgb_model, rf_pipeline, xgb_pipeline, 
                                                 rf_importance, xgb_importance, i)
        
        # Save patient description as JSON
        with open(os.path.join(output_dir, f"patient_{i:04d}_description.json"), 'w') as f:
            json.dump(patient_desc, f, indent=2)
        
        # Generate and save LLM prompt
        prompt = generate_llm_prompt(patient_desc)
        with open(os.path.join(output_dir, f"patient_{i:04d}_prompt.txt"), 'w') as f:
            f.write(prompt)
    
    print(f"Generated {num_patients} patient descriptions and prompts in {output_dir}")

if __name__ == "__main__":
    main() 