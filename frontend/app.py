import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD
model = joblib.load("../models/best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
encoder = joblib.load("../models/encoder.pkl")

df = pd.read_csv("../data/meteorite_cleaned.csv")

st.title("🌠 Meteorite Landing Analysis")

# DATA
st.subheader("Dataset")
st.dataframe(df.head())

# EDA
st.subheader("EDA")
st.image("../data/mass_distribution.png")
st.image("../data/mass_vs_year.png")
st.image("../data/top_classes.png")
st.image("../data/avg_mass.png")

# PREDICTION
st.subheader("Predict Mass")

year = st.slider("Year", 1900, 2025, 2000)
recclass = st.selectbox("Class", df['recclass'].unique())

if st.button("Predict"):
year_scaled = year - df['year'].min()
class_encoded = encoder.transform([recclass])[0]

X = np.array([[year_scaled, class_encoded]])
X_scaled = scaler.transform(X)

pred = model.predict(X_scaled)
mass = np.expm1(pred)

st.success(f"Predicted Mass: {mass[0]:.2f} kg")