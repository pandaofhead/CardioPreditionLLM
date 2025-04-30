# CVD Risk Prediction Project

This project provides cardiovascular disease (CVD) risk prediction and explanation using machine learning models and large language models (LLMs).

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r app/requirements.txt
   ```
3. Create a `.env.local` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the App

Run the Streamlit app:

```
cd /path/to/project
streamlit run app/app.py
```

## Acknowledgments

- Heart Disease UCI dataset
- SHAP library for model interpretability
- OpenAI API for generating explanations