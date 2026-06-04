import streamlit as st
import pandas as pd
import numpy as np
import joblib

# NOTE:
# This is a Streamlit application.
# Deploy it on Streamlit Community Cloud, Render, or Railway.
# It is NOT compatible with Vercel's Python runtime without being rewritten as Flask/FastAPI.

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

st.title("🩺 Diabetes Risk Predictor")
st.write("Application loaded successfully.")
