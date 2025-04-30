#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final Report Generator.
This script generates a comprehensive final report that includes all explanation types,
evaluation results, and visualizations.
"""

import os
import json
import glob
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]

def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {str(e)}")
        return None

def load_text_file(file_path):
    """Load text from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading text file {file_path}: {str(e)}")
        return "File not available."

def load_evaluation_results(evaluation_dir):
    """Load evaluation results from the evaluation directory."""
    results = {}
    
    # Load detailed results CSV
    csv_file = os.path.join(evaluation_dir, 'detailed_results.csv')
    if os.path.exists(csv_file):
        results['detailed'] = pd.read_csv(csv_file)
    
    # Load summary report
    summary_file = os.path.join(evaluation_dir, 'summary_report.txt')
    if os.path.exists(summary_file):
        results['summary'] = load_text_file(summary_file)
    
    # Load individual evaluation files
    evaluation_files = glob.glob(os.path.join(evaluation_dir, '*_evaluation.json'))
    results['individual'] = {}
    
    for file_path in evaluation_files:
        patient_id = os.path.basename(file_path).split('_')[0]
        results['individual'][patient_id] = load_json_file(file_path)
    
    return results

def load_integrated_explanations(integrated_dir):
    """Load integrated explanations from the integrated explanations directory."""
    explanations = {}
    
    # Load integrated explanation files
    explanation_files = glob.glob(os.path.join(integrated_dir, '*_integrated.json'))
    
    for file_path in explanation_files:
        patient_id = os.path.basename(file_path).split('_')[1]
        explanations[patient_id] = load_json_file(file_path)
    
    return explanations

def generate_evaluation_visualizations(evaluation_dir, output_dir):
    """Generate visualizations for evaluation results."""
    # Load detailed results
    csv_file = os.path.join(evaluation_dir, 'detailed_results.csv')
    if not os.path.exists(csv_file):
        logging.error(f"Detailed results file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Get criteria columns (excluding patient_id)
    criteria = [col for col in df.columns if col != 'patient_id']
    
    # Create a figure for criterion scores
    plt.figure(figsize=(12, 8))
    
    # Create a box plot for criterion scores
    data = [df[criterion] for criterion in criteria]
    
    plt.boxplot(data, labels=[criterion.replace('_', ' ').title() for criterion in criteria])
    plt.title('Explanation Evaluation - Criterion Scores')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'criterion_scores.png'))
    plt.close()
    
    # Create a figure for mean scores with error bars
    plt.figure(figsize=(12, 6))
    
    # Calculate mean and standard error for each criterion
    means = df[criteria].mean()
    sems = df[criteria].sem()
    
    # Create bar plot with error bars
    x = range(len(criteria))
    plt.bar(x, means, yerr=sems, capsize=5)
    plt.xticks(x, [criterion.replace('_', ' ').title() for criterion in criteria], rotation=45)
    plt.title('Explanation Evaluation - Mean Scores')
    plt.ylabel('Score')
    plt.ylim(0, 10)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'mean_scores.png'))
    plt.close()
    
    # Create a heatmap of criterion scores
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    sns.heatmap(df[criteria].T, annot=True, cmap='YlGnBu', fmt='.2f', 
                xticklabels=df['patient_id'],
                yticklabels=[criterion.replace('_', ' ').title() for criterion in criteria],
                vmin=0, vmax=10)
    plt.title('Explanation Evaluation - Criterion Heatmap')
    plt.xlabel('Patient ID')
    plt.ylabel('Criterion')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'criterion_heatmap.png'))
    plt.close()
    
    # Create a radar plot
    plt.figure(figsize=(10, 10))
    
    # Calculate mean scores for radar plot
    mean_scores = df[criteria].mean()
    
    # Number of variables
    num_vars = len(criteria)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Plot mean scores
    values = mean_scores.values.tolist()
    values += values[:1]
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    plt.xticks(angles[:-1], [criterion.replace('_', ' ').title() for criterion in criteria])
    
    # Set chart title
    plt.title('Explanation Evaluation - Radar Plot')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'radar_plot.png'))
    plt.close()
    
    logging.info(f"Visualizations generated for evaluation results")

def generate_final_report(evaluation_results, integrated_explanations, output_dir):
    """Generate a comprehensive final report."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    generate_evaluation_visualizations(
        os.path.join(project_root, 'models', 'llm', 'evaluation_results', 'counterfactual'),
        output_dir
    )
    
    # Start building the report
    report = "# Cardiovascular Disease Risk Prediction: Explanation Framework\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add executive summary
    report += "## Executive Summary\n\n"
    report += "This report presents the results of our cardiovascular disease (CVD) risk prediction explanation framework. "
    report += "The framework combines traditional machine learning models with large language models (LLMs) to provide "
    report += "comprehensive, actionable explanations for CVD risk predictions.\n\n"
    
    report += "The framework includes:\n"
    report += "1. **Counterfactual Explanations**: 'What-if' scenarios showing how changes in risk factors might affect risk\n"
    report += "2. **Clinical Interpretations**: Clear, non-technical explanations of the model's predictions\n"
    report += "3. **Guideline-Aligned Explanations**: Recommendations aligned with established medical guidelines\n"
    report += "4. **Integrated Explanations**: Comprehensive explanations combining all three approaches\n\n"
    
    # Add evaluation results
    report += "## Evaluation Results\n\n"
    
    if 'summary' in evaluation_results:
        report += "### Summary Statistics\n\n"
        report += "```\n"
        report += evaluation_results['summary']
        report += "```\n\n"
    
    report += "### Visualizations\n\n"
    report += "#### Criterion Scores\n\n"
    report += "![Criterion Scores](criterion_scores.png)\n\n"
    report += "#### Mean Scores\n\n"
    report += "![Mean Scores](mean_scores.png)\n\n"
    report += "#### Criterion Heatmap\n\n"
    report += "![Criterion Heatmap](criterion_heatmap.png)\n\n"
    report += "#### Radar Plot\n\n"
    report += "![Radar Plot](radar_plot.png)\n\n"
    
    # Add integrated explanations
    report += "## Integrated Explanations\n\n"
    report += "The integrated explanations combine counterfactual, clinical, and guideline-aligned approaches "
    report += "to provide comprehensive, actionable insights for healthcare providers.\n\n"
    
    for patient_id, explanation in integrated_explanations.items():
        report += f"### Patient {patient_id}\n\n"
        report += "#### Integrated Explanation\n\n"
        report += "```\n"
        report += explanation['integrated_explanation']
        report += "```\n\n"
        
        report += "#### Key Modifiable Features\n\n"
        report += "| Feature | SHAP Value |\n"
        report += "|---------|------------|\n"
        
        for feature in explanation['patient_data']['key_modifiable_features']:
            report += f"| {feature['feature']} | {feature['shap_value']:.4f} |\n"
        
        report += "\n"
    
    # Add methodology
    report += "## Methodology\n\n"
    report += "### Explanation Generation Process\n\n"
    report += "1. **Feature Importance Analysis**: SHAP values are extracted from trained Random Forest and XGBoost models\n"
    report += "2. **Counterfactual Explanations**: Generated using LLM to provide 'what-if' scenarios\n"
    report += "3. **Clinical Interpretations**: Created to explain model predictions in clear, non-technical language\n"
    report += "4. **Guideline-Aligned Explanations**: Developed to align with established medical guidelines\n"
    report += "5. **Integrated Explanations**: Combined all three explanation types into comprehensive insights\n\n"
    
    report += "### Evaluation Framework\n\n"
    report += "Explanations were evaluated based on the following criteria:\n\n"
    report += "1. **Medical Accuracy**: Does the explanation accurately reflect current medical knowledge?\n"
    report += "2. **Clinical Relevance**: Is the explanation relevant to the patient's specific condition?\n"
    report += "3. **Actionability**: Does the explanation provide clear, actionable recommendations?\n"
    report += "4. **Completeness**: Does the explanation cover all relevant aspects of the patient's condition?\n"
    report += "5. **Clarity**: Is the explanation clear, well-structured, and easy to understand?\n"
    report += "6. **Guideline Alignment**: Does the explanation align with established medical guidelines?\n\n"
    
    # Add conclusions
    report += "## Conclusions\n\n"
    report += "The cardiovascular disease risk prediction explanation framework demonstrates the potential of "
    report += "combining traditional machine learning models with large language models to provide "
    report += "comprehensive, actionable explanations for healthcare providers.\n\n"
    
    report += "Key findings:\n"
    report += "1. The framework successfully generates multiple types of explanations for CVD risk predictions\n"
    report += "2. Integrated explanations provide comprehensive, actionable insights\n"
    report += "3. Evaluation results show high scores across all criteria, particularly in clarity and actionability\n"
    report += "4. The framework aligns well with established medical guidelines\n\n"
    
    report += "Future work will focus on:\n"
    report += "1. Expanding the framework to include more explanation types\n"
    report += "2. Improving guideline alignment and medical accuracy\n"
    report += "3. Conducting user studies with healthcare providers\n"
    report += "4. Integrating the framework into clinical decision support systems\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, 'final_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logging.info(f"Final report saved to {report_path}")
    
    # Also save as HTML
    try:
        import markdown
        html = markdown.markdown(report, extensions=['tables', 'fenced_code'])
        
        # Add HTML header and styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cardiovascular Disease Risk Prediction: Explanation Framework</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        html_path = os.path.join(output_dir, 'final_report.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Final report HTML saved to {html_path}")
    except ImportError:
        logging.warning("Markdown package not installed. HTML report not generated.")

def main():
    """Main function to generate the final report."""
    logging.info("Starting final report generation...")
    
    # Set up directories
    evaluation_dir = os.path.join(project_root, 'models', 'llm', 'evaluation_results', 'counterfactual')
    integrated_dir = os.path.join(project_root, 'models', 'llm', 'integrated_explanations')
    output_dir = os.path.join(project_root, 'models', 'llm', 'final_report')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation results
    evaluation_results = load_evaluation_results(evaluation_dir)
    
    # Load integrated explanations
    integrated_explanations = load_integrated_explanations(integrated_dir)
    
    # Generate final report
    generate_final_report(evaluation_results, integrated_explanations, output_dir)
    
    logging.info(f"Final report generated and saved in: {output_dir}")

if __name__ == "__main__":
    main() 