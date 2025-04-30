#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP analysis for model interpretability.
This script performs SHAP analysis on trained Random Forest and XGBoost models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]

def load_models():
    """Load trained Random Forest and XGBoost models"""
    rf_pipeline = joblib.load(os.path.join(project_root, 'models', 'traditional_ml', 'random_forest.pkl'))
    xgb_pipeline = joblib.load(os.path.join(project_root, 'models', 'traditional_ml', 'xgboost.pkl'))
    # Extract the actual models from the pipelines
    rf_model = rf_pipeline.named_steps['classifier']
    xgb_model = xgb_pipeline.named_steps['classifier']
    return rf_model, xgb_model, rf_pipeline, xgb_pipeline

def load_validation_data():
    """Load validation dataset for SHAP analysis"""
    data_dir = os.path.join(project_root, 'data', 'processed')
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    return X_val

def perform_shap_analysis(model, pipeline, X_val, model_name, output_dir):
    """Perform SHAP analysis for a given model"""
    print(f"\nGenerating SHAP analysis for {model_name}...")
    
    # Transform the data using the pipeline's preprocessing steps
    X_val_transformed = pipeline.named_steps['scaler'].transform(X_val)
    X_val_transformed = pd.DataFrame(X_val_transformed, columns=X_val.columns)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_transformed)
    
    # For binary classification, we want the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot SHAP summary (bar plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val_transformed, feature_names=X_val.columns.tolist(), 
                     max_display=20, plot_type='bar')
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_bar.png'))
    plt.close()
    
    # Plot SHAP summary (scatter plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val_transformed, feature_names=X_val.columns.tolist(), 
                     max_display=20)
    plt.title(f'SHAP Summary Plot - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'))
    plt.close()
    
    # Calculate and save feature importance
    feature_importance = np.zeros(X_val.shape[1])
    for i in range(X_val.shape[1]):
        feature_importance[i] = np.mean(np.abs(shap_values[:, i]))
    
    importance_df = pd.DataFrame({
        'feature': X_val.columns.tolist(),
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.csv'), 
                        index=False)
    
    return importance_df

def main():
    """Main function to run SHAP analysis"""
    print("Starting SHAP analysis...")
    
    # Load models and data
    rf_model, xgb_model, rf_pipeline, xgb_pipeline = load_models()
    X_val = load_validation_data()
    
    # Set output directory
    output_dir = os.path.join(project_root, 'models', 'traditional_ml', 'feature_importance')
    
    # Perform SHAP analysis for both models
    rf_importance = perform_shap_analysis(rf_model, rf_pipeline, X_val, "Random Forest", output_dir)
    xgb_importance = perform_shap_analysis(xgb_model, xgb_pipeline, X_val, "XGBoost", output_dir)
    
    # Print top features
    print("\nTop 10 important features for Random Forest:")
    print(rf_importance.head(10))
    print("\nTop 10 important features for XGBoost:")
    print(xgb_importance.head(10))
    
    print("\nSHAP analysis completed. Results saved in:", output_dir)

if __name__ == "__main__":
    main() 