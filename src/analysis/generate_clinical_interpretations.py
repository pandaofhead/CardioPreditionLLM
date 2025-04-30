#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clinical Interpretation Generator using LLM API.
This script loads patient descriptions and generates clinical interpretations using LLM.
"""

import os
import json
import time
from pathlib import Path
import openai
from dotenv import load_dotenv
from tqdm import tqdm

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

def process_patient_descriptions(input_dir, output_dir):
    """Process all patient descriptions and generate clinical interpretations"""
    # Load API key
    api_key = load_env()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all prompt files
    prompt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_prompt.txt')])
    
    print(f"Found {len(prompt_files)} patient prompts to process")
    
    # Process each prompt
    for prompt_file in tqdm(prompt_files, desc="Generating interpretations"):
        patient_id = prompt_file.split('_')[1]  # Extract patient ID from filename
        
        # Read prompt
        with open(os.path.join(input_dir, prompt_file), 'r') as f:
            prompt = f.read()
        
        try:
            # Get LLM interpretation
            interpretation = get_llm_response(api_key, prompt)
            
            # Save interpretation
            output_file = f"patient_{patient_id}_interpretation.txt"
            with open(os.path.join(output_dir, output_file), 'w') as f:
                f.write(interpretation)
            
            # Also save as part of a structured JSON
            result = {
                "patient_id": f"P{patient_id}",
                "prompt": prompt,
                "interpretation": interpretation
            }
            
            json_output_file = f"patient_{patient_id}_result.json"
            with open(os.path.join(output_dir, json_output_file), 'w') as f:
                json.dump(result, f, indent=2)
            
            # Add a small delay between requests to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
            continue

def main():
    """Main function to generate clinical interpretations"""
    print("Starting clinical interpretation generation...")
    
    # Set up directories
    input_dir = os.path.join(project_root, 'models', 'llm', 'patient_descriptions')
    output_dir = os.path.join(project_root, 'models', 'llm', 'clinical_interpretations')
    
    # Process patient descriptions
    process_patient_descriptions(input_dir, output_dir)
    
    print(f"\nClinical interpretations generated and saved in: {output_dir}")

if __name__ == "__main__":
    main() 