#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit demo application for the CVD Risk Prediction project.
This app displays patient data and pre-generated LLM explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import markdown

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Set page config
st.set_page_config(
    page_title="CVD Risk Prediction & Explanation Framework",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B77BE;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #E6F3FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .high-risk {
        color: #FF4B4B;
        font-weight: bold;
    }
    .moderate-risk {
        color: #FFA500;
        font-weight: bold;
    }
    .low-risk {
        color: #4CAF50;
        font-weight: bold;
    }
    .explanation-box {
        background-color: #F9F9F9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4B77BE;
    }
    .evaluation-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .framework-box {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)


def load_integrated_explanations():
    """
    Load integrated explanations from the integrated explanations directory
    
    Returns:
    --------
    dict
        Dictionary of integrated explanations
    """
    integrated_dir = os.path.join(project_root, 'models', 'llm', 'integrated_explanations')
    
    if not os.path.exists(integrated_dir):
        st.warning("Integrated explanations directory not found.")
        return {}
    
    explanations = {}
    
    # Load integrated explanation files
    explanation_files = glob.glob(os.path.join(integrated_dir, '*_integrated.json'))
    
    for file_path in explanation_files:
        try:
            with open(file_path, 'r') as f:
                explanation = json.load(f)
                patient_id = os.path.basename(file_path).split('_')[1]
                explanations[patient_id] = explanation
        except Exception as e:
            st.error(f"Error loading explanation from {file_path}: {str(e)}")
    
    return explanations


def load_evaluation_results():
    """
    Load evaluation results from the evaluation directory
    
    Returns:
    --------
    dict
        Dictionary of evaluation results
    """
    evaluation_dir = os.path.join(project_root, 'models', 'llm', 'evaluation_results', 'counterfactual')
    
    if not os.path.exists(evaluation_dir):
        st.warning("Evaluation results directory not found.")
        return {}
    
    results = {}
    
    # Load detailed results CSV
    csv_file = os.path.join(evaluation_dir, 'detailed_results.csv')
    if os.path.exists(csv_file):
        try:
            results['detailed'] = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Error loading detailed results: {str(e)}")
    
    # Load summary report
    summary_file = os.path.join(evaluation_dir, 'summary_report.txt')
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                results['summary'] = f.read()
        except Exception as e:
            st.error(f"Error loading summary report: {str(e)}")
    
    # Load visualization files
    results['visualizations'] = {}
    viz_files = ['criterion_scores.png', 'mean_scores.png', 'criterion_heatmap.png', 'radar_plot.png']
    
    for viz_file in viz_files:
        file_path = os.path.join(evaluation_dir, f"counterfactual_{viz_file}")
        if os.path.exists(file_path):
            results['visualizations'][viz_file] = file_path
    
    return results


def load_final_report():
    """
    Load the final report
    
    Returns:
    --------
    str
        The final report content
    """
    report_dir = os.path.join(project_root, 'models', 'llm', 'final_report')
    
    if not os.path.exists(report_dir):
        st.warning("Final report directory not found.")
        return None
    
    report_file = os.path.join(report_dir, 'final_report.md')
    
    if not os.path.exists(report_file):
        st.warning("Final report file not found.")
        return None
    
    try:
        with open(report_file, 'r') as f:
            report_content = f.read()
        return report_content
    except Exception as e:
        st.error(f"Error loading final report: {str(e)}")
        return None


def display_explanation_framework():
    """
    Display the explanation framework section
    """
    st.markdown('<h2 class="sub-header">Explanation Framework</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="framework-box">
    <p>Our cardiovascular disease risk prediction framework combines traditional machine learning models with large language models (LLMs) to provide comprehensive, actionable explanations for CVD risk predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the framework components
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h3>Framework Components</h3>
        <ol>
            <li><strong>Feature Importance Analysis</strong>: SHAP values extracted from trained Random Forest and XGBoost models</li>
            <li><strong>Counterfactual Explanations</strong>: 'What-if' scenarios showing how changes in risk factors might affect risk</li>
            <li><strong>Clinical Interpretations</strong>: Clear, non-technical explanations of the model's predictions</li>
            <li><strong>Guideline-Aligned Explanations</strong>: Recommendations aligned with established medical guidelines</li>
            <li><strong>Integrated Explanations</strong>: Comprehensive explanations combining all three approaches</li>
        </ol>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h3>Evaluation Framework</h3>
        <p>Explanations are evaluated based on the following criteria:</p>
        <ul>
            <li><strong>Medical Accuracy</strong>: Does the explanation accurately reflect current medical knowledge?</li>
            <li><strong>Clinical Relevance</strong>: Is the explanation relevant to the patient's specific condition?</li>
            <li><strong>Actionability</strong>: Does the explanation provide clear, actionable recommendations?</li>
            <li><strong>Completeness</strong>: Does the explanation cover all relevant aspects of the patient's condition?</li>
            <li><strong>Clarity</strong>: Is the explanation clear, well-structured, and easy to understand?</li>
            <li><strong>Guideline Alignment</strong>: Does the explanation align with established medical guidelines?</li>
        </ul>
        """, unsafe_allow_html=True)


def display_evaluation_results(evaluation_results):
    """
    Display the evaluation results section
    
    Parameters:
    -----------
    evaluation_results : dict
        Dictionary of evaluation results
    """
    st.markdown('<h2 class="sub-header">Evaluation Results</h2>', unsafe_allow_html=True)
    
    if not evaluation_results:
        st.warning("No evaluation results available.")
        return
    
    # Display summary statistics
    if 'summary' in evaluation_results:
        st.markdown("""
        <div class="evaluation-box">
        <h3>Summary Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.text(evaluation_results['summary'])
    
    # Display visualizations
    if 'visualizations' in evaluation_results and evaluation_results['visualizations']:
        st.markdown("""
        <div class="evaluation-box">
        <h3>Visualizations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display visualizations in a grid
        cols = st.columns(2)
        
        for i, (viz_name, viz_path) in enumerate(evaluation_results['visualizations'].items()):
            with cols[i % 2]:
                st.image(viz_path, caption=viz_name.replace('_', ' ').title())
    
    # Display detailed results
    if 'detailed' in evaluation_results and isinstance(evaluation_results['detailed'], pd.DataFrame):
        st.markdown("""
        <div class="evaluation-box">
        <h3>Detailed Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(evaluation_results['detailed'])


def display_integrated_explanations(explanations):
    """
    Display the integrated explanations section
    
    Parameters:
    -----------
    explanations : dict
        Dictionary of integrated explanations
    """
    st.markdown('<h2 class="sub-header">Integrated Explanations</h2>', unsafe_allow_html=True)
    
    if not explanations:
        st.warning("No integrated explanations available.")
        return
    
    st.markdown("""
    <div class="explanation-box">
    <p>The integrated explanations combine counterfactual, clinical, and guideline-aligned approaches to provide comprehensive, actionable insights for healthcare providers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a dropdown to select a patient
    patient_ids = list(explanations.keys())
    selected_patient = st.selectbox("Select a patient:", sorted(patient_ids), key="integrated_patient_select")
    
    if selected_patient:
        explanation = explanations[selected_patient]
        
        st.markdown(f"""
        <div class="explanation-box">
        <h3>Patient {selected_patient}</h3>
        <h4>Integrated Explanation</h4>
        <p>{explanation.get('integrated_explanation', '')}</p>
        </h4>Key Modifiable Features</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display key modifiable features in a table if available
        if 'patient_data' in explanation and 'key_modifiable_features' in explanation['patient_data']:
            features = explanation['patient_data']['key_modifiable_features']
            
            if features:
                feature_data = []
                for feature in features:
                    feature_data.append({
                        'Feature': feature['feature'],
                        'SHAP Value': f"{feature['shap_value']:.4f}"
                    })
                
                st.table(pd.DataFrame(feature_data))
            else:
                st.info("No key modifiable features available for this patient.")


def display_final_report(report_content):
    """
    Display the final report section
    
    Parameters:
    -----------
    report_content : str
        The final report content
    """
    st.markdown('<h2 class="sub-header">Final Report</h2>', unsafe_allow_html=True)
    
    if not report_content:
        st.warning("Final report not available.")
        return
    
    # Convert markdown to HTML
    try:
        html_content = markdown.markdown(report_content, extensions=['tables', 'fenced_code'])
        # Display the report
        st.markdown(html_content, unsafe_allow_html=True)
    except:
        # Fallback to displaying as markdown if conversion fails
        st.markdown(report_content)


def load_explanation_by_type(explanation_type, patient_id):
    """
    Load a specific explanation type for a given patient
    
    Parameters:
    -----------
    explanation_type : str
        Type of explanation (counterfactual, clinical, guideline_aligned)
    patient_id : str
        ID of the patient
    
    Returns:
    --------
    dict or None
        The explanation data if found, None otherwise
    """
    explanation_dir = os.path.join(project_root, 'models', 'llm', f'{explanation_type}_explanations')
    
    if not os.path.exists(explanation_dir):
        return None
    
    file_pattern = f"patient_{patient_id}_{explanation_type.replace('_explanations', '')}"
    # Try TXT file if JSON not found or failed
    txt_file = os.path.join(explanation_dir, f"{file_pattern}.txt")
    if os.path.exists(txt_file):
        try:
            with open(txt_file, 'r') as f:
                return {"text_content": f.read()}
        except Exception as e:
            st.error(f"Error loading explanation from {txt_file}: {str(e)}")
    # Try JSON file first
    json_file = os.path.join(explanation_dir, f"{file_pattern}.json")
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading explanation from {json_file}: {str(e)}")
    
    return None


def display_individual_explanations():
    """
    Display the individual explanation types (counterfactual, clinical, guideline-aligned)
    """
    st.markdown('<h2 class="sub-header">Individual Explanations</h2>', unsafe_allow_html=True)
    
    # Get list of available patients
    patient_ids = []
    for dir_name in ['counterfactual_explanations', 'clinical_interpretations', 'guideline_aligned_explanations']:
        dir_path = os.path.join(project_root, 'models', 'llm', dir_name)
        if os.path.exists(dir_path):
            for file_path in glob.glob(os.path.join(dir_path, '*.json')):
                patient_id = os.path.basename(file_path).split('_')[1]
                if patient_id not in patient_ids:
                    patient_ids.append(patient_id)
    
    if not patient_ids:
        st.warning("No explanations available.")
        return
    
    # Create a dropdown to select a patient
    selected_patient = st.selectbox("Select a patient:", sorted(patient_ids), key="individual_patient_select")
    
    if selected_patient:
        # Create tabs for different explanation types
        clinical_tab, counterfactual_tab, guideline_tab = st.tabs([
            "Clinical Interpretation", 
            "Counterfactual Explanation",
            "Guideline-Aligned Explanation"
        ])
        
        # Display clinical interpretation
        with clinical_tab:
            explanation = load_explanation_by_type("clinical", selected_patient)
            
            # Also try with _interpretation suffix if not found
            if explanation is None:
                explanation = load_explanation_by_type("clinical_interpretation", selected_patient)
            
            # Check in result.json file
            if explanation is None:
                result_file = os.path.join(project_root, 'models', 'llm', 'clinical_interpretations', f'patient_{selected_patient}_result.json')
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            explanation = json.load(f)
                    except Exception as e:
                        st.error(f"Error loading clinical interpretation: {str(e)}")
            
            # Check in interpretation.txt file
            if explanation is None or "interpretation" not in explanation:
                interp_file = os.path.join(project_root, 'models', 'llm', 'clinical_interpretations', f'patient_{selected_patient}_interpretation.txt')
                if os.path.exists(interp_file):
                    try:
                        with open(interp_file, 'r') as f:
                            explanation = {"text_content": f.read()}
                    except Exception as e:
                        st.error(f"Error loading clinical interpretation: {str(e)}")
            
            if explanation:
                if "text_content" in explanation:
                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>Clinical Interpretation</h3>
                    <p>{explanation['text_content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif "interpretation" in explanation:
                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>Clinical Interpretation</h3>
                    <p>{explanation['interpretation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.json(explanation)
            else:
                st.info("No clinical interpretation available for this patient.")
        # Display counterfactual explanation
        with counterfactual_tab:
            explanation = load_explanation_by_type("counterfactual", selected_patient)
            
            if explanation:
                if "text_content" in explanation:
                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>Counterfactual Explanation</h3>
                    <p>{explanation['text_content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif "explanation" in explanation:
                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>Counterfactual Explanation</h3>
                    <p>{explanation['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.json(explanation)
            else:
                st.info("No counterfactual explanation available for this patient.")
        
        # Display guideline-aligned explanation
        with guideline_tab:
            explanation = load_explanation_by_type("guideline_aligned", selected_patient)
            if explanation:
                if "text_content" in explanation:
                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>Guideline-Aligned Explanation</h3>
                    <p>{explanation['text_content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif "explanation" in explanation:
                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>Guideline-Aligned Explanation</h3>
                    <p>{explanation['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.json(explanation)
            else:
                st.info("No guideline-aligned explanation available for this patient.")


def display_patient_descriptions():
    """
    Display the patient descriptions
    """
    st.markdown('<h2 class="sub-header">Patient Descriptions</h2>', unsafe_allow_html=True)
    
    # Get list of available patients
    patient_ids = []
    desc_dir = os.path.join(project_root, 'models', 'llm', 'patient_descriptions')
    if os.path.exists(desc_dir):
        for file_path in glob.glob(os.path.join(desc_dir, '*_description.json')):
            patient_id = os.path.basename(file_path).split('_')[1]
            if patient_id not in patient_ids:
                patient_ids.append(patient_id)
    
    if not patient_ids:
        st.warning("No patient descriptions available.")
        return
    
    # Create a dropdown to select a patient
    selected_patient = st.selectbox("Select a patient:", sorted(patient_ids), key="desc_patient_select")
    
    if selected_patient:
        # Load patient description
        desc_file = os.path.join(desc_dir, f"patient_{selected_patient}_description.json")
        prompt_file = os.path.join(desc_dir, f"patient_{selected_patient}_prompt.txt")
        
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'r') as f:
                    description = json.load(f)
                
                # Display basic patient info
                st.markdown(f"""
                <div class="info-box">
                <h3>Patient {selected_patient} Information</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Parse the patient data based on the actual JSON structure
                cols = st.columns(2)
                
                with cols[0]:
                    st.subheader("Demographics")
                    if 'demographics' in description:
                        demo = description['demographics']
                        demo_data = {
                            "Age": demo.get('age', 'N/A'),
                            "Sex": demo.get('sex', 'N/A')
                        }
                        st.table(pd.DataFrame(list(demo_data.items()), columns=["Feature", "Value"]))
                    
                    st.subheader("Vital Signs")
                    if 'clinical_features' in description:
                        cf = description['clinical_features']
                        vital_data = {
                            "Resting Blood Pressure": f"{cf.get('resting_blood_pressure', 'N/A')} mm Hg",
                            "Serum Cholesterol": f"{cf.get('serum_cholesterol', 'N/A')} mg/dl",
                            "Fasting Blood Sugar": cf.get('fasting_blood_sugar', 'N/A'),
                            "Maximum Heart Rate": cf.get('max_heart_rate', 'N/A')
                        }
                        st.table(pd.DataFrame(list(vital_data.items()), columns=["Feature", "Value"]))
                
                with cols[1]:
                    st.subheader("Cardiac Assessment")
                    if 'clinical_features' in description:
                        cf = description['clinical_features']
                        cardiac_data = {
                            "Chest Pain Type": cf.get('chest_pain_type', 'N/A'),
                            "Resting ECG": cf.get('resting_ecg', 'N/A'),
                            "Exercise Induced Angina": cf.get('exercise_induced_angina', 'N/A'),
                            "ST Depression": cf.get('st_depression', 'N/A'),
                            "ST Segment Slope": cf.get('st_slope', 'N/A'),
                            "Number of Vessels": cf.get('number_of_vessels', 'N/A'),
                            "Thalassemia": cf.get('thalassemia', 'N/A')
                        }
                        st.table(pd.DataFrame(list(cardiac_data.items()), columns=["Feature", "Value"]))
                
                # Display model predictions
                if 'predictions' in description:
                    st.subheader("Model Predictions")
                    preds = description['predictions']
                    
                    model_tabs = st.tabs(["Random Forest", "XGBoost"])
                    
                    # Random Forest prediction
                    with model_tabs[0]:
                        if 'random_forest' in preds:
                            rf = preds['random_forest']
                            st.markdown(f"""
                            <div class="prediction-box">
                                <p><strong>Prediction:</strong> {rf.get('prediction', 'N/A')}</p>
                                <p><strong>Confidence:</strong> {rf.get('confidence', 'N/A'):.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display SHAP values
                            if 'shap_values' in rf:
                                rf_shap = rf['shap_values']
                                rf_shap_df = pd.DataFrame({
                                    'Feature': list(rf_shap.keys()),
                                    'SHAP Value': list(rf_shap.values())
                                }).sort_values('SHAP Value', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='SHAP Value', y='Feature', data=rf_shap_df, ax=ax)
                                ax.set_title('Random Forest SHAP Values')
                                st.pyplot(fig)
                    
                    # XGBoost prediction
                    with model_tabs[1]:
                        if 'xgboost' in preds:
                            xgb = preds['xgboost']
                            st.markdown(f"""
                            <div class="prediction-box">
                                <p><strong>Prediction:</strong> {xgb.get('prediction', 'N/A')}</p>
                                <p><strong>Confidence:</strong> {xgb.get('confidence', 'N/A'):.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display SHAP values
                            if 'shap_values' in xgb:
                                xgb_shap = xgb['shap_values']
                                xgb_shap_df = pd.DataFrame({
                                    'Feature': list(xgb_shap.keys()),
                                    'SHAP Value': list(xgb_shap.values())
                                }).sort_values('SHAP Value', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='SHAP Value', y='Feature', data=xgb_shap_df, ax=ax)
                                ax.set_title('XGBoost SHAP Values')
                                st.pyplot(fig)
                
                # Display prompt if available
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        prompt_content = f.read()
                    
                    with st.expander("View LLM Prompt"):
                        st.code(prompt_content, language="markdown")
                
            except Exception as e:
                st.error(f"Error loading patient description: {str(e)}")
        else:
            st.info(f"No description file found for patient {selected_patient}")


def main():
    """
    Main function to run the Streamlit app
    """
    st.markdown('<h1 class="main-header">Cardiovascular Disease Risk Prediction & Explanation Framework</h1>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Patient Descriptions",
        "Individual Explanations", 
        "Integrated Explanations",
        "Evaluation Results"
    ])
    
    with tab1:
        display_patient_descriptions()
    
    with tab2:
        display_individual_explanations()
    
    with tab3:
        # Load and display integrated explanations
        explanations = load_integrated_explanations()
        display_integrated_explanations(explanations)
    
    with tab4:
        # Load and display evaluation results
        evaluation_results = load_evaluation_results()
        display_evaluation_results(evaluation_results)
    
    # Add footnotes and references
    st.markdown("""
    <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc; font-size: 0.8em;">
        <p><strong>References:</strong></p>
        <ol>
            <li>Heart Disease Dataset: UCI Machine Learning Repository</li>
            <li>American Heart Association Guidelines for CVD Risk Assessment</li>
            <li>Framingham Heart Study Risk Assessment Model</li>
        </ol>
        <p><strong>Disclaimer:</strong> This application is for educational purposes only and should not be used for medical diagnosis or treatment. Always consult with a healthcare professional for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 