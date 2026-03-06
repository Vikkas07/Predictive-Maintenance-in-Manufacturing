import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import MODEL_PATH, FEATURE_COLUMNS

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["threshold"], bundle["columns"]

def prepare_input(data_dict):
    df = pd.DataFrame([data_dict])
    df = df[FEATURE_COLUMNS]
    return df

def predict_failure(model, input_df):
    prob = model.predict_proba(input_df)[:,1][0]
    return float(prob)

def maintenance_decision(prob, threshold):
    if prob >= threshold:
        return "MAINTENANCE REQUIRED", "red"
    elif prob >= threshold * 0.6:
        return "INSPECTION ADVISED", "orange"
    else:
        return "NORMAL OPERATION", "green"
