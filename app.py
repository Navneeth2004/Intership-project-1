import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/region_encoder.pkl")

st.set_page_config(page_title="Customer Predictor", layout="wide")

st.title("🛒 Customer Purchasing Behavior Prediction")

# ======================
# INPUT SECTION
# ======================
st.header("🔧 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income", value=50000)

with col2:
    purchase_amount = st.number_input("Purchase Amount", value=200)
    loyalty_score = st.slider("Loyalty Score", 1.0, 10.0, 5.0)

region = st.selectbox("Region", encoder.classes_)

# ======================
# PREDICTION BUTTON
# ======================
if st.button("🚀 Predict Purchase Frequency"):
    
    region_encoded = encoder.transform([region])[0]

    input_data = np.array([[age, income, purchase_amount, loyalty_score, region_encoded]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"📊 Predicted Purchase Frequency: {prediction:.2f}")

# ======================
# VISUALIZATION SECTION
# ======================
st.header("📊 Data Insights")

df = pd.read_csv("data/Customer Purchasing Behaviors.csv")

col3, col4 = st.columns(2)

# Histogram
with col3:
    fig1, ax1 = plt.subplots()
    df["purchase_frequency"].hist(bins=20)
    ax1.set_title("Purchase Frequency Distribution")
    st.pyplot(fig1)

# Feature Visualization
with col4:
    fig2, ax2 = plt.subplots()
    sample = df.sample(50)
    ax2.scatter(sample["annual_income"], sample["purchase_frequency"])
    ax2.set_xlabel("Income")
    ax2.set_ylabel("Purchase Frequency")
    ax2.set_title("Income vs Purchase Frequency")
    st.pyplot(fig2)