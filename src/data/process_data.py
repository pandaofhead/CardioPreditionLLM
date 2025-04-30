#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing script for the CVD Risk Prediction project.
This script loads the raw heart disease dataset, performs exploratory data analysis,
preprocesses the data, and splits it into train, validation, and test sets.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def load_data(data_path=None):
    """
    Load the raw heart disease dataset
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the dataset file. If None, uses the default path.
    
    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'raw', 'heart.csv')
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def check_missing_values(df):
    """
    Check for missing values in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to check
    
    Returns:
    --------
    pandas.DataFrame
        A dataframe showing the missing values count and percentage
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.concat([missing, missing_percent], axis=1)
    missing_data.columns = ['Total', 'Percent']
    return missing_data[missing_data['Total'] > 0]


def analyze_data(df, save_path=None):
    """
    Perform exploratory data analysis and save visualizations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    save_path : str, optional
        Path to save the visualizations. If None, uses the default path.
    
    Returns:
    --------
    dict
        A dictionary containing the analysis results
    """
    if save_path is None:
        save_path = os.path.join(project_root, 'docs', 'figures')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Target distribution
    target_counts = df['target'].value_counts()
    print("\nTarget Distribution:")
    print(target_counts)
    print(f"Positive class ratio: {target_counts[1] / len(df):.4f}")
    
    # Create visualizations
    # 1. Target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Heart Disease Distribution')
    plt.xlabel('Target (0 = No Disease, 1 = Disease)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_path, 'target_distribution.png'))
    plt.close()
    
    # 2. Age distribution by target
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=30)
    plt.title('Age Distribution by Heart Disease')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_path, 'age_distribution.png'))
    plt.close()
    
    # 3. Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    plt.close()
    
    # 4. Sex distribution by target
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sex', hue='target', data=df)
    plt.title('Heart Disease by Sex')
    plt.xlabel('Sex (0 = Female, 1 = Male)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_path, 'sex_distribution.png'))
    plt.close()
    
    # 5. Chest pain type by target
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cp', hue='target', data=df)
    plt.title('Heart Disease by Chest Pain Type')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_path, 'chest_pain_distribution.png'))
    plt.close()
    
    # 6. Boxplots for continuous variables
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig, axes = plt.subplots(len(continuous_features), 1, figsize=(12, 20))
    fig.tight_layout(pad=5.0)
    
    for i, feature in enumerate(continuous_features):
        sns.boxplot(x='target', y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'{feature} by Heart Disease')
        axes[i].set_xlabel('Heart Disease')
        axes[i].set_ylabel(feature)
    
    plt.savefig(os.path.join(save_path, 'continuous_features_boxplots.png'))
    plt.close()
    
    return {
        'target_distribution': target_counts,
        'correlation_matrix': correlation_matrix
    }


def preprocess_data(df):
    """
    Preprocess the data for modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to preprocess
    
    Returns:
    --------
    dict
        A dictionary containing the preprocessed data splits
    """
    # Create a copy of the dataframe
    processed_df = df.copy()
    
    # Handle missing values (if any)
    processed_df = processed_df.dropna()
    
    # Separate features and target
    X = processed_df.drop('target', axis=1)
    y = processed_df['target']
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split sizes:")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframes with column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Save the scaler
    models_dir = os.path.join(project_root, 'models', 'traditional_ml')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create dataset splits dictionary
    dataset_splits = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled_df,
        'X_val_scaled': X_val_scaled_df,
        'X_test_scaled': X_test_scaled_df,
        'scaler': scaler
    }
    
    return dataset_splits


def save_processed_data(dataset_splits, output_dir=None):
    """
    Save the processed data
    
    Parameters:
    -----------
    dataset_splits : dict
        Dictionary containing the data splits
    output_dir : str, optional
        Directory to save the processed data. If None, uses the default path.
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, 'data', 'processed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    dataset_splits['X_train'].to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    dataset_splits['y_train'].to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    dataset_splits['X_val'].to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    dataset_splits['y_val'].to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    dataset_splits['X_test'].to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    dataset_splits['y_test'].to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save scaled versions
    dataset_splits['X_train_scaled'].to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    dataset_splits['X_val_scaled'].to_csv(os.path.join(output_dir, 'X_val_scaled.csv'), index=False)
    dataset_splits['X_test_scaled'].to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    
    # Save complete processed data
    processed_df = pd.concat([
        pd.concat([dataset_splits['X_train'], dataset_splits['y_train']], axis=1),
        pd.concat([dataset_splits['X_val'], dataset_splits['y_val']], axis=1),
        pd.concat([dataset_splits['X_test'], dataset_splits['y_test']], axis=1)
    ])
    processed_df.to_csv(os.path.join(output_dir, 'heart_processed.csv'), index=False)
    
    # Save splits as pickle for convenience
    with open(os.path.join(output_dir, 'dataset_splits.pkl'), 'wb') as f:
        # Remove the scaler as it's saved separately
        splits_to_save = {k: v for k, v in dataset_splits.items() if k != 'scaler'}
        pickle.dump(splits_to_save, f)
    
    print(f"Processed data saved to {output_dir}")


def main():
    """
    Main function to process the heart disease dataset
    """
    # Create necessary directories
    docs_dir = os.path.join(project_root, 'docs', 'figures')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Starting data processing for CVD Risk Prediction project...")
    
    # Load data
    df = load_data()
    
    # Check for missing values
    missing_data = check_missing_values(df)
    print("\nMissing values:")
    print(missing_data if not missing_data.empty else "No missing values found")
    
    # Analyze data
    analyze_data(df)
    
    # Preprocess data
    dataset_splits = preprocess_data(df)
    
    # Save processed data
    save_processed_data(dataset_splits)
    
    print("\nData processing complete!")


if __name__ == "__main__":
    main() 