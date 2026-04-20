import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Meteorite Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# -----------------------------
# UI THEME (NEW)
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }

    h1, h2, h3 {
        color: #00d4ff;
    }

    .stSidebar {
        background-color: #111827;
    }

    div.stButton > button {
        background-color: #00d4ff;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #00aacc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "backend", "data", "processed", "meteorite_final.csv")
ARIMA_PATH = os.path.join(BASE_DIR, "backend", "models", "arima_model.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# -----------------------------
# HEADER
# -----------------------------
st.title("☄️ Meteorite Analytics Dashboard")
st.markdown("Explore meteorite landings, trends, ML prediction & forecasting.")

st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["📊 EDA", "🤖 Prediction (ML)", "📈 ARIMA Forecasting"]
)

# =====================================================
# 📊 EDA
# =====================================================
if menu == "📊 EDA":

    st.header("Exploratory Data Analysis")

    # ---------------- CLEANING ----------------
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mass_tonnes"] = pd.to_numeric(df["mass_tonnes"], errors="coerce")
    df = df.dropna(subset=["year", "mass_tonnes"])
    df["year"] = df["year"].astype(int)
    df = df[df["mass_tonnes"] > 0]

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- MASS ----------------
    st.subheader("Mass Distribution (Log Scale)")
    fig1, ax1 = plt.subplots()
    ax1.hist(np.log10(df["mass_tonnes"] + 1), bins=30)
    st.pyplot(fig1)

    # ---------------- YEAR TREND ----------------
    st.subheader("Meteorite Trend Over Years")
    yearly = df.groupby("year").size().rolling(3, min_periods=1).mean()

    fig2, ax2 = plt.subplots()
    ax2.plot(yearly.index, yearly.values)
    st.pyplot(fig2)

    # ---------------- MASS VS YEAR ----------------
    st.subheader("Mass vs Year")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df["year"], np.log1p(df["mass_tonnes"]), alpha=0.5)
    st.pyplot(fig3)

    # ---------------- HEATMAP ----------------
    st.subheader("Correlation Heatmap")

    df_heat = df.copy()
    df_heat["class_code"] = df["class"].astype("category").cat.codes
    df_heat["country_code"] = df["country"].astype("category").cat.codes

    corr = df_heat[["year", "mass_tonnes", "class_code", "country_code"]].corr()

    fig4, ax4 = plt.subplots()
    cax = ax4.imshow(corr, cmap="coolwarm")
    fig4.colorbar(cax)

    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, rotation=45)
    ax4.set_yticklabels(corr.columns)

    st.pyplot(fig4)

    # ---------------- BUBBLE ----------------
    st.subheader("Bubble Chart")

    bubble = df.groupby("year").agg({
        "mass_tonnes": "mean",
        "name": "count"
    }).reset_index()

    fig5, ax5 = plt.subplots()
    ax5.scatter(
        bubble["year"],
        bubble["mass_tonnes"],
        s=bubble["name"] * 20,
        alpha=0.6
    )
    st.pyplot(fig5)

    # ---------------- BULLET (FIXED) ----------------
    st.subheader("Bullet Chart")

    avg_mass = df["mass_tonnes"].mean()
    max_mass = df["mass_tonnes"].max()

    fig6, ax6 = plt.subplots(figsize=(6,2))
    ax6.barh(["Mass"], [max_mass], color="lightgray")
    ax6.barh(["Mass"], [avg_mass], color="steelblue")
    ax6.set_xlim(0, max_mass * 1.1)

    st.pyplot(fig6)

    # ---------------- MICRO TREND (FIXED) ----------------
    st.subheader("Micro Trend (Top Regions)")

    top_regions = df["region"].dropna().value_counts().head(5).index

    fig7, ax7 = plt.subplots()

    plotted = False

    for region in top_regions:
        sub = df[df["region"] == region]

        if sub["year"].nunique() < 2:
            continue

        trend = sub.groupby("year").size().sort_index()
        trend = trend.rolling(2, min_periods=1).mean()

        ax7.plot(trend.index, trend.values, label=region)
        plotted = True

    if plotted:
        ax7.legend()

    st.pyplot(fig7)

# =====================================================
# 🤖 ML PREDICTION
# =====================================================
elif menu == "🤖 Prediction (ML)":

    st.header("Meteorite Mass Prediction")

    try:
        model = joblib.load(os.path.join(BASE_DIR, "backend", "models", "mass_model.pkl"))
        encoders = joblib.load(os.path.join(BASE_DIR, "backend", "models", "encoders.pkl"))

        year = st.number_input("Year", 1800, 2100, 2000)
        country = st.text_input("Country", "India")
        region = st.text_input("Region", "Unknown")
        meteorite_class = st.text_input("Class", "H5")

        def safe_encode(col, val):
            if val in encoders[col].classes_:
                return encoders[col].transform([val])[0]
            return -1

        if st.button("Predict Mass"):

            X_input = np.array([[
                year,
                safe_encode("country", country),
                safe_encode("region", region),
                safe_encode("class", meteorite_class)
            ]])

            pred = model.predict(X_input)[0]
            pred = max(pred, 0)

            st.success(f"Predicted Mass: {pred:.4f} tonnes")

    except Exception as e:
        st.error(f"Model error: {e}")

# =====================================================
# 📈 ARIMA
# =====================================================
elif menu == "📈 ARIMA Forecasting":

    st.header("Meteorite Forecasting")

    try:
        arima_model = joblib.load(ARIMA_PATH)
        series = joblib.load(os.path.join(BASE_DIR, "backend", "models", "year_series.pkl"))

        steps = st.slider("Forecast Years", 1, 10, 5)

        forecast = arima_model.predict(n_periods=steps)
        forecast = np.clip(forecast, 0, None)
        forecast = np.round(forecast).astype(int)

        last_year = int(series.index[-1])
        future_years = np.arange(last_year + 1, last_year + steps + 1)

        fig, ax = plt.subplots()

        ax.plot(series.index, series.values, label="Actual")
        ax.plot(future_years, forecast, marker="o", linestyle="dashed", label="Forecast")

        ax.legend()
        st.pyplot(fig)

        st.subheader("Forecast Values")
        for y, v in zip(future_years, forecast):
            st.write(f"{y} → {v}")

    except Exception as e:
        st.error(f"ARIMA error: {e}")