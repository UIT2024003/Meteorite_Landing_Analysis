import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD
model = joblib.load("../models/meteorite_model.pkl")
year_min = joblib.load("../models/year_min.pkl")

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

if st.button("Predict"):
    # scale year same as training
    year_scaled = year - year_min

    X = np.array([[year_scaled]])   # only 1 feature

    pred = model.predict(X)
    mass = np.expm1(pred)

    st.success(f"Predicted Mass: {mass[0]:.2f} kg")