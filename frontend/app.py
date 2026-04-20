import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Meteorite Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# -----------------------------
# UI THEME
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
    return pd.read_csv(DATA_PATH)

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

    # -----------------------------
    # CLEAN DATA (SAFE)
    # -----------------------------
    df_eda = df.copy()

    df_eda["year"] = pd.to_numeric(df_eda["year"], errors="coerce")
    df_eda["mass_tonnes"] = pd.to_numeric(df_eda["mass_tonnes"], errors="coerce")

    # FIX lat/lon
    if "lat" in df_eda.columns:
        df_eda["lat"] = pd.to_numeric(df_eda["lat"], errors="coerce")
    else:
        df_eda["lat"] = np.nan

    if "lon" in df_eda.columns:
        df_eda["lon"] = pd.to_numeric(df_eda["lon"], errors="coerce")
    else:
        df_eda["lon"] = np.nan

    df_eda = df_eda.dropna(subset=["year", "mass_tonnes"])

    df_eda = df_eda[
        (df_eda["year"] > 1800) &
        (df_eda["year"] < 2025)
    ]

    # ⚠️ IMPORTANT FIX (no over-filtering)
    df_eda = df_eda[df_eda["mass_tonnes"] > 0]

    df_eda["class"] = df_eda["class"].fillna("Unknown")

    if df_eda.empty:
        st.error("No data available after cleaning")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df_eda.head())

    # =====================================================
    # 1. MASS DISTRIBUTION
    # =====================================================
    st.subheader("Mass Distribution (Log Scale)")

    fig, ax = plt.subplots()

    mass_log = np.log10(df_eda["mass_tonnes"] + 1e-9)

    ax.hist(mass_log, bins=50)

    ax.set_xlabel("Log10(Mass)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # =====================================================
    # 2. YEAR TREND
    # =====================================================
    st.subheader("Meteorite Count Over Years")

    yearly = df_eda.groupby("year").size()

    fig, ax = plt.subplots()
    ax.plot(yearly.index, yearly.values)

    st.pyplot(fig)

    # =====================================================
    # 3. MASS VS YEAR
    # =====================================================
    st.subheader("Mass vs Year")

    sample_df = df_eda.sample(min(5000, len(df_eda)))

    fig, ax = plt.subplots()

    ax.scatter(
        sample_df["year"],
        np.log10(sample_df["mass_tonnes"] + 1e-9),
        alpha=0.4
    )

    st.pyplot(fig)

    # =====================================================
    # 4. TOP CLASSES
    # =====================================================
    st.subheader("Top Meteorite Classes")

    fig, ax = plt.subplots()
    df_eda["class"].value_counts().head(10).plot(kind="bar", ax=ax)

    st.pyplot(fig)

    # =====================================================
    # 5. MEDIAN MASS BY CLASS
    # =====================================================
    st.subheader("Median Mass by Class")

    class_mass = (
        df_eda.groupby("class")["mass_tonnes"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    class_mass.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    # =====================================================
    # 6. MAP (FIXED)
    # =====================================================
    st.subheader("Meteorite Locations")

    map_df = df_eda.dropna(subset=["lat", "lon"])

    if len(map_df) > 0:
        st.map(map_df.rename(columns={
            "lat": "latitude",
            "lon": "longitude"
        }))
    else:
        st.warning("No location data available")

  
# =====================================================
# 🤖 ML PREDICTION
# =====================================================
# =====================================================
# 🤖 ML PREDICTION
# =====================================================
elif menu == "🤖 Prediction (ML)":

    st.header("Meteorite Class Prediction")

    try:
        @st.cache_resource
        def load_model():
            return joblib.load(os.path.join(BASE_DIR, "backend", "models", "class_model.pkl"))

        model = load_model()

        year = st.number_input("Year", 1800, 2025, 2000)
        mass = st.number_input("Mass (tonnes)", 0.000001, 10.0, 0.01)
        lat = st.number_input("Latitude", -90.0, 90.0, 20.0)
        lon = st.number_input("Longitude", -180.0, 180.0, 70.0)

        if st.button("Predict Class"):

            input_df = pd.DataFrame([{
                "year": year,
                "mass_tonnes": mass,
                "lat": lat,
                "lon": lon
            }])

            # 🔥 Probability-based prediction
            probs = model.predict_proba(input_df)[0]
            classes = model.classes_

            top_idx = probs.argmax()

            st.success(f"Predicted Class: {classes[top_idx]}")
            st.write(f"Confidence: {probs[top_idx]*100:.2f}%")

            # 🔥 Top 3 predictions (very useful)
            st.subheader("Top 3 Possible Classes")

            top_3_idx = probs.argsort()[-3:][::-1]

            for i in top_3_idx:
                st.write(f"{classes[i]} → {probs[i]*100:.2f}%")
            avg_mass = df[df["class"] == classes[top_idx]]["mass_tonnes"].mean()
            st.write(f"Average mass for {classes[top_idx]}: {avg_mass:.4f} tonnes")
            st.bar_chart(pd.DataFrame({
    "Class": classes[top_3_idx],
    "Probability": probs[top_3_idx]
}).set_index("Class"))
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