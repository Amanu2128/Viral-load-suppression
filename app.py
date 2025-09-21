# app.py - Full Functional Version with All Variables
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Page config
st.set_page_config(page_title="VLS Prediction App", page_icon="🧬", layout="wide")

# -----------------------------
# 2. Header
st.title("🧬 HIV Viral Load Suppression Predictor")
st.markdown("Predict Viral Load Suppression using ML models developed for your PhD research.")

# -----------------------------
# 3. Load Model & Components
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgboost_f1_optimized_model.joblib")
        scaler = joblib.load("scaler.joblib")
        features = joblib.load("model_features.joblib")
        threshold = joblib.load("xgboost_f1_optimized_threshold.joblib")
        return model, scaler, features, threshold
    except Exception as e:
        st.error(f"Model files missing or corrupted: {e}")
        st.stop()

model, scaler, all_features, best_threshold = load_model()

NUMERIC_SCALER_COLS = ['Age', 'CD4 Count', 'HIV infection LAg ODN test']
MINORITY_CLASS_LABEL = 1

# -----------------------------
# 4. Sidebar Info
st.sidebar.header("ℹ️ Project Info")
st.sidebar.markdown(
    """
    This app predicts **Viral Load Suppression** (VLS) for HIV patients on ART.
    
    **Model:** Optimized XGBoost  
    **Threshold Optimization:** F1 Score (Minority Class)  
    **Developer:** Your PhD Project
    """
)

# -----------------------------
# 5. Feature Inputs
st.subheader("📊 Patient Inputs")
input_data = {}

col1, col2 = st.columns(2)

with col1:
    input_data['Age'] = st.slider("Age (Years)", 15, 80, 35)
    input_data['CD4 Count'] = st.slider("CD4 Count (cells/mm³)", 50, 1000, 500)
    input_data['HIV infection LAg ODN test'] = st.slider("HIV Infection LAg ODN Test", 0.0, 10.0, 3.0, 0.1)
    input_data['ART Duration'] = st.selectbox("ART Duration (Months)", ['<12 Months', '12-23 Months', '>=24 Months'])
    input_data['Sick Last 3 Months'] = st.selectbox("Sick in Last 3 Months?", ['No', 'Yes'])
    input_data['TB Clinic Visit'] = st.selectbox("TB Clinic Visit in Last 12 Months?", ['No', 'Yes'])
    input_data['Wealth Quintile'] = st.selectbox("Wealth Quintile", ['Poorest', 'Poor', 'Middle', 'Rich', 'Richest', 'Unknown'])
    input_data['Paid Work'] = st.selectbox("Engaged in Paid Work?", ['No', 'Yes'])
    input_data['Residence Type'] = st.selectbox("Residence Type", ['Rural', 'Urban'])

with col2:
    input_data['Gender'] = st.selectbox("Gender", ['Female', 'Male'])
    input_data['Educational Status'] = st.selectbox("Education Level", ['No Schooling', 'Primary', 'Secondary', 'Higher', 'Unknown'])
    input_data['Marital Status'] = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Widowed', 'Cohabiting', 'Separated', 'Unknown'])
    input_data['HIV Status Disclosure'] = st.selectbox("HIV Status Disclosure", ['Disclosed', 'Not Disclosed', 'Unknown'])
    input_data['Efavirenz Drug'] = st.selectbox("Taking Efavirenz Drug?", ['No', 'Yes'])
    input_data['Lopinavir Drug'] = st.selectbox("Taking Lopinavir Drug?", ['No', 'Yes'])
    input_data['Nevirapine Drug'] = st.selectbox("Taking Nevirapine Drug?", ['No', 'Yes'])
    input_data['Condom Use'] = st.selectbox("Condom Use in Last Sexual Activity?", ['No', 'Yes'])
    input_data['Sex in Last 12 Months'] = st.selectbox("Had Sex in Last 12 Months?", ['No', 'Yes'])

# -----------------------------
# Map string selections to numeric
mapping_dict = {
    'Gender': {'Female':0, 'Male':1},
    'ART Duration': {'<12 Months':0, '12-23 Months':1, '>=24 Months':2},
    'Sick Last 3 Months': {'No':0, 'Yes':1},
    'TB Clinic Visit': {'No':0, 'Yes':1},
    'Wealth Quintile': {'Poorest':0,'Poor':1,'Middle':2,'Rich':3,'Richest':4,'Unknown':5},
    'Paid Work': {'No':0,'Yes':1},
    'Residence Type': {'Rural':0,'Urban':1},
    'Educational Status': {'No Schooling':0, 'Primary':1, 'Secondary':2, 'Higher':3, 'Unknown':4},
    'Marital Status': {'Single':0,'Married':1,'Divorced':2,'Widowed':3,'Cohabiting':4,'Separated':5,'Unknown':6},
    'HIV Status Disclosure': {'Disclosed':0,'Not Disclosed':1,'Unknown':2},
    'Efavirenz Drug': {'No':0,'Yes':1},
    'Lopinavir Drug': {'No':0,'Yes':1},
    'Nevirapine Drug': {'No':0,'Yes':1},
    'Condom Use': {'No':0,'Yes':1},
    'Sex in Last 12 Months': {'No':0,'Yes':1},
}

for k, v in mapping_dict.items():
    input_data[k] = v[input_data[k]]

# -----------------------------
# Ensure all_features exist
for f in all_features:
    if f not in input_data and f != 'Viral Load Suppression':
        input_data[f] = 0

# -----------------------------
# 6. Prediction
st.subheader("🚀 Generate Prediction")
if st.button("Predict VLS"):
    df_input = pd.DataFrame([input_data])[ [f for f in all_features if f != 'Viral Load Suppression'] ]

    # Scale numeric features
    if NUMERIC_SCALER_COLS:
        df_input[NUMERIC_SCALER_COLS] = scaler.transform(df_input[NUMERIC_SCALER_COLS])

    # Predict
    proba = model.predict_proba(df_input)[:, MINORITY_CLASS_LABEL][0]
    final_pred = int(proba >= best_threshold)

    # Display
    if final_pred == MINORITY_CLASS_LABEL:
        st.error(f"❌ Viral Load: UNSUPPRESSED\nProbability: {proba:.2f}")
        st.info("High risk detected. Recommend clinical follow-up.")
    else:
        st.success(f"✅ Viral Load: SUPPRESSED\nProbability: {proba:.2f}")
        st.info("Positive outcome predicted. Continue standard monitoring.")
