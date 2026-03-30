import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit import button

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)


st.title("Random Forest Demonstration")

st.sidebar.header("Transaction Features")

V1 = st.sidebar.slider("V1", -30.0, 30.0, 0.0)
V2 = st.sidebar.slider("V2", -30.0, 30.0, 0.0)
V3 = st.sidebar.slider("V3", -30.0, 30.0, 0.0)
V4 = st.sidebar.slider("V4", -30.0, 30.0, 0.0)
V5 = st.sidebar.slider("V5", -30.0, 30.0, 0.0)
V6 = st.sidebar.slider("V6", -30.0, 30.0, 0.0)
V7 = st.sidebar.slider("V7", -30.0, 30.0, 0.0)
V8 = st.sidebar.slider("V8", -30.0, 30.0, 0.0)
V9 = st.sidebar.slider("V9", -30.0, 30.0, 0.0)
V10 = st.sidebar.slider("V10", -30.0, 30.0, 0.0)
V11 = st.sidebar.slider("V11", -30.0, 30.0, 0.0)
V12 = st.sidebar.slider("V12", -30.0, 30.0, 0.0)
V13 = st.sidebar.slider("V13", -30.0, 30.0, 0.0)
V14 = st.sidebar.slider("V14", -30.0, 30.0, 0.0)
V15 = st.sidebar.slider("V15", -30.0, 30.0, 0.0)
V16 = st.sidebar.slider("V16", -30.0, 30.0, 0.0)
V17 = st.sidebar.slider("V17", -30.0, 30.0, 0.0)
V18 = st.sidebar.slider("V18", -30.0, 30.0, 0.0)
V19 = st.sidebar.slider("V19", -30.0, 30.0, 0.0)
V20 = st.sidebar.slider("V20", -30.0, 30.0, 0.0)
V21 = st.sidebar.slider("V21", -30.0, 30.0, 0.0)
V22 = st.sidebar.slider("V22", -30.0, 30.0, 0.0)
V23 = st.sidebar.slider("V23", -30.0, 30.0, 0.0)
V24 = st.sidebar.slider("V24", -30.0, 30.0, 0.0)
V25 = st.sidebar.slider("V25", -30.0, 30.0, 0.0)
V26 = st.sidebar.slider("V26", -30.0, 30.0, 0.0)
V27 = st.sidebar.slider("V27", -30.0, 30.0, 0.0)
V28 = st.sidebar.slider("V28", -30.0, 30.0, 0.0)
Hour = st.sidebar.slider("Hour", 0.0, 23.0, 12.0)
is_night = st.sidebar.selectbox("Is Night?", [0, 1])
Time_scaled = st.sidebar.slider("Time_scaled", -5.0, 5.0, 0.0)
Amount_scaled = st.sidebar.slider("Amount_scaled", -5.0, 5.0, 0.0)

if st.button("Predict"):
    data = pd.DataFrame([[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                          V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
                          V21, V22, V23, V24, V25, V26, V27, V28,
                          Hour, is_night, Time_scaled, Amount_scaled]],
                        columns=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                                 "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                                 "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                                 "Hour", "is_night", "Time_scaled", "Amount_scaled"])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud probability: {probability:.2%}")
