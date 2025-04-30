#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Explanation Generator using LLM API.
This script combines different types of explanations (counterfactual, clinical, guideline-aligned)
into a comprehensive, integrated explanation for each patient.
"""

import os
import json
import time
import logging
from pathlib import Path
import glob
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]

def load_env():
    """Load environment variables from .env.local file"""
    env_path = os.path.join(project_root, '.env.local')
    load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return api_key

def get_llm_response(prompt, max_retries=3, delay=2):
    """Get response from OpenAI API with retry logic."""
    api_key = load_env()
    
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant helping to interpret cardiovascular disease risk predictions. Provide clear, accurate, and actionable medical insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to get LLM response after {max_retries} attempts: {str(e)}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)

def load_explanation(file_path):
    """Load explanation from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading explanation from {file_path}: {str(e)}")
        return None

def generate_integrated_explanation(patient_id, counterfactual_explanation, clinical_interpretation, guideline_aligned_explanation, patient_data):
    """Generate an integrated explanation combining all explanation types."""
    prompt = f"""You are a medical AI assistant helping to interpret cardiovascular disease (CVD) risk predictions.

Patient Information:
{json.dumps(patient_data, indent=2)}

I have three different types of explanations for this patient's CVD risk:

1. COUNTERFACTUAL EXPLANATION:
{counterfactual_explanation}

2. CLINICAL INTERPRETATION:
{clinical_interpretation}

3. GUIDELINE-ALIGNED EXPLANATION:
{guideline_aligned_explanation}

Please create an integrated, comprehensive explanation that:
1. Combines the strengths of all three explanation types
2. Eliminates redundancies
3. Provides a clear, structured, and actionable explanation
4. Follows a logical flow from risk assessment to recommendations
5. Is written in clear, non-technical language that a healthcare provider could use to explain the situation to a patient

Your integrated explanation should include:
1. A summary of the patient's current risk status
2. Key risk factors and their contributions
3. Specific, actionable recommendations
4. Expected outcomes if recommendations are followed
5. References to relevant clinical guidelines where appropriate

Format your response as a well-structured medical explanation with clear sections and bullet points where appropriate.
"""
    
    return get_llm_response(prompt)

def process_patient_explanations(patient_id, counterfactual_dir, clinical_dir, guideline_dir, output_dir):
    """Process explanations for a single patient and generate an integrated explanation."""
    # Load counterfactual explanation
    counterfactual_file = os.path.join(counterfactual_dir, f"patient_{patient_id}_counterfactual.json")
    counterfactual_data = load_explanation(counterfactual_file)
    if not counterfactual_data:
        logging.error(f"Could not load counterfactual explanation for patient {patient_id}")
        return
    
    # Load clinical interpretation
    clinical_file = os.path.join(clinical_dir, f"patient_{patient_id}_interpretation.txt")
    try:
        with open(clinical_file, 'r') as f:
            clinical_interpretation = f.read()
    except Exception as e:
        logging.error(f"Could not load clinical interpretation for patient {patient_id}: {str(e)}")
        clinical_interpretation = "Clinical interpretation not available."
    
    # Load guideline-aligned explanation
    guideline_file = os.path.join(guideline_dir, f"patient_{patient_id}_guideline_aligned.txt")
    try:
        with open(guideline_file, 'r') as f:
            guideline_aligned_explanation = f.read()
    except Exception as e:
        logging.error(f"Could not load guideline-aligned explanation for patient {patient_id}: {str(e)}")
        guideline_aligned_explanation = "Guideline-aligned explanation not available."
    
    # Extract patient data
    patient_data = {
        'patient_id': counterfactual_data['patient_id'],
        'key_modifiable_features': counterfactual_data['key_modifiable_features']
    }
    
    # Generate integrated explanation
    logging.info(f"Generating integrated explanation for patient {patient_id}")
    integrated_explanation = generate_integrated_explanation(
        patient_id,
        counterfactual_data['counterfactual_explanation'],
        clinical_interpretation,
        guideline_aligned_explanation,
        patient_data
    )
    
    # Save integrated explanation
    result = {
        'patient_id': f"P{patient_id}",
        'counterfactual_explanation': counterfactual_data['counterfactual_explanation'],
        'clinical_interpretation': clinical_interpretation,
        'guideline_aligned_explanation': guideline_aligned_explanation,
        'integrated_explanation': integrated_explanation,
        'patient_data': patient_data
    }
    
    output_file = os.path.join(output_dir, f"patient_{patient_id}_integrated.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Also save the integrated explanation as a separate text file
    text_output_file = os.path.join(output_dir, f"patient_{patient_id}_integrated.txt")
    with open(text_output_file, 'w') as f:
        f.write(integrated_explanation)
    
    logging.info(f"Integrated explanation saved for patient {patient_id}")
    
    return result

def main():
    """Main function to generate integrated explanations."""
    logging.info("Starting integrated explanation generation...")
    
    # Set up directories
    counterfactual_dir = os.path.join(project_root, 'models', 'llm', 'counterfactual_explanations')
    clinical_dir = os.path.join(project_root, 'models', 'llm', 'clinical_interpretations')
    guideline_dir = os.path.join(project_root, 'models', 'llm', 'guideline_aligned_explanations')
    output_dir = os.path.join(project_root, 'models', 'llm', 'integrated_explanations')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all counterfactual explanation files
    counterfactual_files = glob.glob(os.path.join(counterfactual_dir, '*_counterfactual.json'))
    patient_ids = [os.path.basename(f).split('_')[1] for f in counterfactual_files]
    
    logging.info(f"Found {len(patient_ids)} patients to process")
    
    # Process each patient
    all_results = {}
    for patient_id in tqdm(patient_ids, desc="Generating integrated explanations"):
        result = process_patient_explanations(
            patient_id,
            counterfactual_dir,
            clinical_dir,
            guideline_dir,
            output_dir
        )
        if result:
            all_results[patient_id] = result
        
        # Add a small delay between requests to respect rate limits
        time.sleep(1)
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    logging.info(f"Integrated explanations generated and saved in: {output_dir}")

def generate_summary_report(results, output_dir):
    """Generate a summary report of all integrated explanations."""
    report = "# Integrated Explanation Generation Summary\n\n"
    report += f"Generated integrated explanations for {len(results)} patients.\n\n"
    
    report += "## Patient IDs\n\n"
    for patient_id in sorted(results.keys()):
        report += f"- Patient {patient_id}\n"
    
    report += "\n## Files Generated\n\n"
    report += "For each patient, the following files were generated:\n"
    report += "- `patient_{id}_integrated.json`: Complete data including all explanation types\n"
    report += "- `patient_{id}_integrated.txt`: Integrated explanation text only\n\n"
    
    report += "## Integration Process\n\n"
    report += "The integrated explanations combine:\n"
    report += "1. Counterfactual explanations\n"
    report += "2. Clinical interpretations\n"
    report += "3. Guideline-aligned explanations\n\n"
    
    report += "The integration process:\n"
    report += "- Combines the strengths of all three explanation types\n"
    report += "- Eliminates redundancies\n"
    report += "- Provides a clear, structured, and actionable explanation\n"
    report += "- Follows a logical flow from risk assessment to recommendations\n"
    report += "- Uses clear, non-technical language\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logging.info(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main() 