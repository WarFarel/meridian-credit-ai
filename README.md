# Meridian Credit - Explainable Default Risk Prototype (COM300)

Machine learning credit-risk prototype and dual-audience explanation system built for Meridian Credit under UK FCA and EU AI Act (Annex III) requirements.

## Project Structure
- `preprocess.py` - Stratified train/test split, missing-value imputation, structural-zero handling, categorical encoding (ordinal + out-of-scale separation), and financial ratio engineering.
- `train.py` - Trains Logistic Regression (baseline) and LightGBM (primary) under a declared 75% validation recall floor, runs subgroup fairness audits, and exports model artifacts.
- `app.py` - Local Gradio web application returning the lending decision, technical SHAP breakdown, and plain-English applicant notice.

## How to Run Locally
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
