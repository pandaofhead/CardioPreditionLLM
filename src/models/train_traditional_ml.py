#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for traditional ML models for the CVD Risk Prediction project.
This script trains and evaluates Logistic Regression, Random Forest, and XGBoost models.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import shap
import joblib
from imblearn.over_sampling import SMOTE

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def load_data(data_dir=None, use_enhanced_features=True):
    """
    Load the processed heart disease dataset
    
    Parameters:
    -----------
    data_dir : str, optional
        Path to the processed data directory. If None, uses the default path.
    use_enhanced_features : bool, default=True
        Whether to use the enhanced features from feature engineering
        
    Returns:
    --------
    dict
        A dictionary containing the data splits for training
    """
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Check for enhanced dataset if requested
    if use_enhanced_features and os.path.exists(os.path.join(data_dir, 'enhanced_dataset_splits.pkl')):
        with open(os.path.join(data_dir, 'enhanced_dataset_splits.pkl'), 'rb') as f:
            dataset_splits = pickle.load(f)
        print(f"Loaded enhanced dataset splits from {data_dir}")
        
        # Use all features version
        dataset_splits['X_train'] = dataset_splits['X_train_with_all_features']
        dataset_splits['X_val'] = dataset_splits['X_val_with_all_features']
        dataset_splits['X_test'] = dataset_splits['X_test_with_all_features']
        
    # Otherwise load the regular dataset splits
    elif os.path.exists(os.path.join(data_dir, 'dataset_splits.pkl')):
        with open(os.path.join(data_dir, 'dataset_splits.pkl'), 'rb') as f:
            dataset_splits = pickle.load(f)
        print(f"Loaded dataset splits from {data_dir}")
        
    else:
        # Load the individual CSV files as a fallback
        print(f"Loading dataset from individual CSV files in {data_dir}")
        dataset_splits = {}
        
        dataset_splits['X_train'] = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        dataset_splits['y_train'] = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['target']
        dataset_splits['X_val'] = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
        dataset_splits['y_val'] = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))['target']
        dataset_splits['X_test'] = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
        dataset_splits['y_test'] = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['target']
    
    return dataset_splits


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to address class imbalance
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    
    Returns:
    --------
    tuple
        (X_train_resampled, y_train_resampled) with balanced classes
    """
    # Check class distribution before SMOTE
    before_counts = pd.Series(y_train).value_counts()
    before_ratio = before_counts[1] / before_counts[0] if 0 in before_counts and 1 in before_counts else 0
    
    print(f"Class distribution before SMOTE:")
    print(before_counts)
    print(f"Positive to negative ratio: {before_ratio:.4f}")
    
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    after_counts = pd.Series(y_resampled).value_counts()
    after_ratio = after_counts[1] / after_counts[0] if 0 in after_counts and 1 in after_counts else 0
    
    print(f"Class distribution after SMOTE:")
    print(after_counts)
    print(f"Positive to negative ratio: {after_ratio:.4f}")
    
    return X_resampled, y_resampled


def evaluate_model(model, X, y, model_name, output_dir=None):
    """
    Evaluate a trained model on the given dataset
    
    Parameters:
    -----------
    model : trained sklearn or xgboost model
        The model to evaluate
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Labels
    model_name : str
        Name of the model for output
    output_dir : str, optional
        Directory to save evaluation results. If None, uses the default path.
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, 'models', 'traditional_ml', 'evaluation')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob)
    }
    
    # Print metrics
    print(f"\n{model_name} evaluation:")
    for metric, value in metrics.items():
        if metric != 'model_name':
            print(f"{metric}: {value:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_roc_curve.png'))
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics


def train_logistic_regression(X_train, y_train, cv=5):
    """
    Train a Logistic Regression model with hyperparameter tuning
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training labels
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        The best trained logistic regression model
    """
    print("\nTraining Logistic Regression model...")
    
    # Define the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    # Create cross-validation object
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        lr, param_grid, cv=cv_obj, 
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_lr = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_lr


def train_random_forest(X_train, y_train, cv=5):
    """
    Train a Random Forest model with hyperparameter tuning
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training labels
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        The best trained random forest model
    """
    print("\nTraining Random Forest model...")
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    # Create cross-validation object
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the model
    rf = RandomForestClassifier(random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv_obj, 
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_rf


def train_xgboost(X_train, y_train, cv=5):
    """
    Train an XGBoost model with hyperparameter tuning
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training labels
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    xgboost.XGBClassifier
        The best trained XGBoost model
    """
    print("\nTraining XGBoost model...")
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]
    }
    
    # Create cross-validation object
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the model
    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=cv_obj, 
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_xgb = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_xgb


def save_model(model, model_name, output_dir=None):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : trained sklearn or xgboost model
        The model to save
    model_name : str
        Name of the model for the filename
    output_dir : str, optional
        Directory to save the model. If None, uses the default path.
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, 'models', 'traditional_ml')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model using joblib
    model_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def create_feature_importance_plots(models, X_val, feature_names, output_dir=None):
    """
    Create feature importance plots for the models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models {'model_name': model}
    X_val : pandas.DataFrame
        Validation features for SHAP analysis
    feature_names : list
        List of feature names
    output_dir : str, optional
        Directory to save the plots. If None, uses the default path.
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, 'models', 'traditional_ml', 'feature_importance')
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\nGenerating feature importance plots for {model_name}...")
        
        if model_name == "Logistic Regression":
            # For Logistic Regression, plot coefficients
            coef = pd.Series(model.coef_[0], index=feature_names)
            coef = coef.sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            coef.head(20).plot(kind='bar')
            plt.title(f'Top 20 Feature Coefficients - {model_name}')
            plt.xlabel('Features')
            plt.ylabel('Coefficient')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_coefficients.png'))
            plt.close()
            
            # Also create SHAP values
            explainer = shap.LinearExplainer(model, X_val)
            shap_values = explainer.shap_values(X_val)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, max_display=20, plot_type='bar')
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_bar.png'))
            plt.close()
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, max_display=20)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'))
            plt.close()
            
        elif model_name == "Random Forest":
            # For Random Forest, plot feature importance
            importances = pd.Series(model.feature_importances_, index=feature_names)
            importances = importances.sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            importances.head(20).plot(kind='bar')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_importance.png'))
            plt.close()
            
            # Create SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[1], X_val, feature_names=feature_names, max_display=20, plot_type='bar')
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_bar.png'))
            plt.close()
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[1], X_val, feature_names=feature_names, max_display=20)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'))
            plt.close()
            
        elif model_name == "XGBoost":
            # For XGBoost, plot feature importance
            plt.figure(figsize=(10, 8))
            xgb.plot_importance(model, max_num_features=20, importance_type='gain')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_importance.png'))
            plt.close()
            
            # Create SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, max_display=20, plot_type='bar')
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_bar.png'))
            plt.close()
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, max_display=20)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'))
            plt.close()


def compare_models(metrics_list, output_dir=None):
    """
    Compare multiple models and create comparison plots
    
    Parameters:
    -----------
    metrics_list : list
        List of dictionaries containing evaluation metrics
    output_dir : str, optional
        Directory to save the comparison plots. If None, uses the default path.
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, 'models', 'traditional_ml', 'comparison')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert list of metrics to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Save comparison metrics
    metrics_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Set up the bar positions
    x = np.arange(len(metrics_df))
    width = 0.15
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i*width, metrics_df[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*2, metrics_df['model_name'])
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # Find the best model
    best_model_idx = metrics_df['roc_auc'].idxmax()
    best_model = metrics_df.loc[best_model_idx, 'model_name']
    best_auc = metrics_df.loc[best_model_idx, 'roc_auc']
    
    print(f"\nBest performing model: {best_model} with ROC AUC = {best_auc:.4f}")
    
    # Save best model info
    with open(os.path.join(output_dir, 'best_model.json'), 'w') as f:
        json.dump({
            'best_model': best_model,
            'metrics': metrics_df.loc[best_model_idx].to_dict()
        }, f, indent=4)
    
    return best_model


def main():
    """
    Main function to train and evaluate traditional ML models
    """
    print("Starting traditional ML model training for CVD Risk Prediction project...")
    
    # Load data
    dataset_splits = load_data(use_enhanced_features=True)
    
    # Get feature names
    feature_names = dataset_splits['X_train'].columns.tolist()
    print(f"Training with {len(feature_names)} features")
    
    # Apply SMOTE for class balancing
    X_train_resampled, y_train_resampled = apply_smote(
        dataset_splits['X_train'], dataset_splits['y_train']
    )
    
    # Train models
    lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)
    xgb_model = train_xgboost(X_train_resampled, y_train_resampled)
    
    # Save models
    save_model(lr_model, "Logistic Regression")
    save_model(rf_model, "Random Forest")
    save_model(xgb_model, "XGBoost")
    
    # Evaluate models on validation set
    lr_metrics = evaluate_model(lr_model, dataset_splits['X_val'], dataset_splits['y_val'], "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, dataset_splits['X_val'], dataset_splits['y_val'], "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, dataset_splits['X_val'], dataset_splits['y_val'], "XGBoost")
    
    # Compare models
    metrics_list = [lr_metrics, rf_metrics, xgb_metrics]
    best_model = compare_models(metrics_list)
    
    # Create feature importance plots
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    create_feature_importance_plots(models, dataset_splits['X_val'], feature_names)
    
    # Evaluate best model on test set
    best_model_obj = models[best_model]
    test_metrics = evaluate_model(
        best_model_obj, 
        dataset_splits['X_test'], 
        dataset_splits['y_test'], 
        f"{best_model} (Test)"
    )
    
    print("\nTraditional ML model training complete!")


if __name__ == "__main__":
    main() 