# import streamlit as st

# # MUST BE FIRST
# st.set_page_config(page_title="Meteorite Dashboard", layout="wide")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import joblib

# # -----------------------------
# # UI THEME
# # -----------------------------
# st.markdown("""
# <style>
#     .main {
#         background-color: #0e1117;
#         color: white;
#     }

#     h1, h2, h3 {
#         color: #00d4ff;
#     }

#     .stSidebar {
#         background-color: #111827;
#     }

#     div.stButton > button {
#         background-color: #00d4ff;
#         color: black;
#         border-radius: 10px;
#         height: 3em;
#         width: 100%;
#     }

#     div.stButton > button:hover {
#         background-color: #00aacc;
#         color: white;
#     }
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------
# # PATHS
# # -----------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA_PATH = os.path.join(BASE_DIR, "backend", "data", "processed", "meteorite_final.csv")
# ARIMA_PATH = os.path.join(BASE_DIR, "backend", "models", "arima_model.pkl")

# # -----------------------------
# # LOAD DATA
# # -----------------------------
# @st.cache_data
# def load_data():
#     return pd.read_csv(DATA_PATH)

# df = load_data()

# # -----------------------------
# # HEADER
# # -----------------------------
# st.title("☄️ Meteorite Analytics Dashboard")
# st.markdown("Explore meteorite landings, trends, ML prediction & forecasting.")

# st.sidebar.title("Navigation")
# menu = st.sidebar.radio(
#     "Select Module",
#     ["📊 EDA", "🤖 Prediction (ML)", "📈 ARIMA Forecasting"]
# )

# # =====================================================
# # 📊 EDA
# # =====================================================
# if menu == "📊 EDA":

#     st.header("Exploratory Data Analysis")

#     # -----------------------------
#     # CLEAN DATA (SAFE)
#     # -----------------------------
#     df_eda = df.copy()

#     df_eda["year"] = pd.to_numeric(df_eda["year"], errors="coerce")
#     df_eda["mass_tonnes"] = pd.to_numeric(df_eda["mass_tonnes"], errors="coerce")

#     # FIX lat/lon
#     if "lat" in df_eda.columns:
#         df_eda["lat"] = pd.to_numeric(df_eda["lat"], errors="coerce")
#     else:
#         df_eda["lat"] = np.nan

#     if "lon" in df_eda.columns:
#         df_eda["lon"] = pd.to_numeric(df_eda["lon"], errors="coerce")
#     else:
#         df_eda["lon"] = np.nan

#     df_eda = df_eda.dropna(subset=["year", "mass_tonnes"])

#     df_eda = df_eda[
#         (df_eda["year"] > 1800) &
#         (df_eda["year"] < 2025)
#     ]

#     # ⚠️ IMPORTANT FIX (no over-filtering)
#     df_eda = df_eda[df_eda["mass_tonnes"] > 0]

#     df_eda["class"] = df_eda["class"].fillna("Unknown")

#     if df_eda.empty:
#         st.error("No data available after cleaning")
#         st.stop()

#     st.subheader("Dataset Preview")
#     st.dataframe(df_eda.head())

#     # =====================================================
#     # 1. MASS DISTRIBUTION
#     # =====================================================
#     st.subheader("Mass Distribution (Log Scale)")

#     fig, ax = plt.subplots()

#     mass_log = np.log10(df_eda["mass_tonnes"] + 1e-9)

#     ax.hist(mass_log, bins=50)

#     ax.set_xlabel("Log10(Mass)")
#     ax.set_ylabel("Frequency")

#     st.pyplot(fig)

#     # =====================================================
#     # 2. YEAR TREND
#     # =====================================================
#     st.subheader("Meteorite Count Over Years")

#     yearly = df_eda.groupby("year").size()

#     fig, ax = plt.subplots()
#     ax.plot(yearly.index, yearly.values)

#     st.pyplot(fig)

#     # =====================================================
#     # 3. MASS VS YEAR
#     # =====================================================
#     st.subheader("Mass vs Year")

#     sample_df = df_eda.sample(min(5000, len(df_eda)))

#     fig, ax = plt.subplots()

#     ax.scatter(
#         sample_df["year"],
#         np.log10(sample_df["mass_tonnes"] + 1e-9),
#         alpha=0.4
#     )

#     st.pyplot(fig)

#     # =====================================================
#     # 4. TOP CLASSES
#     # =====================================================
#     st.subheader("Top Meteorite Classes")

#     fig, ax = plt.subplots()
#     df_eda["class"].value_counts().head(10).plot(kind="bar", ax=ax)

#     st.pyplot(fig)

#     # =====================================================
#     # 5. MEDIAN MASS BY CLASS
#     # =====================================================
#     st.subheader("Median Mass by Class")

#     class_mass = (
#         df_eda.groupby("class")["mass_tonnes"]
#         .median()
#         .sort_values(ascending=False)
#         .head(10)
#     )

#     fig, ax = plt.subplots()
#     class_mass.plot(kind="bar", ax=ax)

#     st.pyplot(fig)

#     # =====================================================
#     # 6. MAP (FIXED)
#     # =====================================================
#     st.subheader("Meteorite Locations")

#     map_df = df_eda.dropna(subset=["lat", "lon"])

#     if len(map_df) > 0:
#         st.map(map_df.rename(columns={
#             "lat": "latitude",
#             "lon": "longitude"
#         }))
#     else:
#         st.warning("No location data available")

  
# # =====================================================
# # 🤖 ML PREDICTION
# # =====================================================

# elif menu == "🤖 Prediction (ML)":

#     st.header("Meteorite Class Prediction")

#     try:
#         @st.cache_resource
#         def load_model():
#             return joblib.load(os.path.join(BASE_DIR, "backend", "models", "class_model.pkl"))

#         model = load_model()

#         year = st.number_input("Year", 1800, 2025, 2000)
#         mass = st.number_input("Mass (tonnes)", 0.000001, 10.0, 0.01)
#         lat = st.number_input("Latitude", -90.0, 90.0, 20.0)
#         lon = st.number_input("Longitude", -180.0, 180.0, 70.0)

#         if st.button("Predict Class"):

#             input_df = pd.DataFrame([{
#                 "year": year,
#                 "mass_tonnes": mass,
#                 "lat": lat,
#                 "lon": lon
#             }])

#             # 🔥 Probability-based prediction
#             probs = model.predict_proba(input_df)[0]
#             classes = model.classes_

#             top_idx = probs.argmax()

#             st.success(f"Predicted Class: {classes[top_idx]}")
#             st.write(f"Confidence: {probs[top_idx]*100:.2f}%")

#             # 🔥 Top 3 predictions (very useful)
#             st.subheader("Top 3 Possible Classes")

#             top_3_idx = probs.argsort()[-3:][::-1]

#             for i in top_3_idx:
#                 st.write(f"{classes[i]} → {probs[i]*100:.2f}%")
#             avg_mass = df[df["class"] == classes[top_idx]]["mass_tonnes"].mean()
#             st.write(f"Average mass for {classes[top_idx]}: {avg_mass:.4f} tonnes")
#             st.bar_chart(pd.DataFrame({
#     "Class": classes[top_3_idx],
#     "Probability": probs[top_3_idx]
# }).set_index("Class"))
#     except Exception as e:
#         st.error(f"Model error: {e}")
        
# # =====================================================
# # 📈 ARIMA
# # =====================================================
# # elif menu == "📈 ARIMA Forecasting":

# #     import matplotlib.pyplot as plt
# #     import joblib
# #     import numpy as np
# #     import os
# #     import streamlit as st

# #     st.header("📈 Meteorite Landing Forecast (SARIMA)")

# #     try:
# #         # -----------------------------
# #         # PATHS
# #         # -----------------------------
# #         BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# #         MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "sarima_model.pkl")
# #         SERIES_PATH = os.path.join(BASE_DIR, "backend", "models", "year_series.pkl")

# #         model = joblib.load(MODEL_PATH)
# #         series = joblib.load(SERIES_PATH)

# #         # -----------------------------
# #         # FILTER HISTORY (IMPORTANT FIX)
# #         # -----------------------------
# #         filtered_series = series[series.index >= 1900]   # 🔥 KEY FIX

# #         # -----------------------------
# #         # FORECAST INPUT
# #         # -----------------------------
# #         years_ahead = st.slider("Forecast Future Years", 1, 20, 10)

# #         # -----------------------------
# #         # FORECAST (log space)
# #         # -----------------------------
# #         forecast_log = model.forecast(steps=years_ahead)
# #         forecast = np.expm1(forecast_log)

# #         # -----------------------------
# #         # FUTURE YEARS
# #         # -----------------------------
# #         last_year = int(filtered_series.index.max())
# #         forecast_years = np.arange(last_year + 1, last_year + years_ahead + 1)

# #         # -----------------------------
# #         # PLOT
# #         # -----------------------------
# #         fig, ax = plt.subplots(figsize=(12, 6))

# #         # Historical
# #         ax.plot(filtered_series.index,
# #                 filtered_series.values,
# #                 label="Historical Data (1900+)",
# #                 color="blue")

# #         # Forecast
# #         ax.plot(forecast_years,
# #                 forecast,
# #                 marker="o",
# #                 linewidth=2,
# #                 label="Forecast",
# #                 color="red")

# #         ax.set_title("Meteorite Mass Forecast (SARIMA)")
# #         ax.set_xlabel("Year")
# #         ax.set_ylabel("Mass (tonnes)")
# #         ax.legend()
# #         ax.grid(True)

# #         # 🔥 FORCE BETTER VISUAL SCALING
# #         ax.set_xlim(1900, forecast_years[-1] + 2)

# #         st.pyplot(fig)

# #         # -----------------------------
# #         # OUTPUT
# #         # -----------------------------
# #         st.subheader("📊 Forecast Values")

# #         for y, val in zip(forecast_years, forecast):
# #             st.write(f"{y} → {round(val, 4)} tonnes")

# #     except Exception as e:
# #         st.error(f"SARIMA error: {e}")
# elif menu == "📈 ARIMA Forecasting":

#     import matplotlib.pyplot as plt
#     import joblib
#     import numpy as np
#     import os
#     import streamlit as st
#     from sklearn.metrics import mean_absolute_error, mean_squared_error
#     from statsmodels.graphics.tsaplots import plot_acf

#     st.header("📈 Meteorite Landing Forecast (SARIMA)")

#     try:
#         # -----------------------------
#         # PATHS
#         # -----------------------------
#         BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#         MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "sarima_model.pkl")
#         SERIES_PATH = os.path.join(BASE_DIR, "backend", "models", "year_series.pkl")

#         model = joblib.load(MODEL_PATH)
#         series = joblib.load(SERIES_PATH)

#         # -----------------------------
#         # FILTER DATA (IMPORTANT FOR VISUALIZATION)
#         # -----------------------------
#         filtered_series = series[series.index >= 1900]

#         # -----------------------------
#         # FORECAST INPUT
#         # -----------------------------
#         years_ahead = st.slider("Forecast Future Years", 1, 20, 10)

#         # -----------------------------
#         # FORECAST (LOG SPACE)
#         # -----------------------------
#         forecast_log = model.forecast(steps=years_ahead)
#         forecast = np.expm1(forecast_log)

#         last_year = int(filtered_series.index.max())
#         forecast_years = np.arange(last_year + 1, last_year + years_ahead + 1)

#         # -----------------------------
#         # PLOT 1: HISTORICAL + FORECAST
#         # -----------------------------
#         fig, ax = plt.subplots(figsize=(12, 6))

#         ax.plot(filtered_series.index,
#                 filtered_series.values,
#                 label="Historical Data (1900+)",
#                 color="blue")

#         ax.plot(forecast_years,
#                 forecast,
#                 marker="o",
#                 linewidth=2,
#                 label="Forecast",
#                 color="red")

#         ax.set_title("Meteorite Mass Forecast (SARIMA)")
#         ax.set_xlabel("Year")
#         ax.set_ylabel("Mass (tonnes)")
#         ax.legend()
#         ax.grid(True)

#         ax.set_xlim(1900, forecast_years[-1] + 2)

#         st.pyplot(fig)

#         # -----------------------------
#         # FORECAST SUMMARY
#         # -----------------------------
#         st.subheader("📊 Forecast Values")

#         for y, val in zip(forecast_years, forecast):
#             st.write(f"{y} → {round(val, 4)} tonnes")

#         st.subheader("📌 Forecast Summary")

#         st.write("📉 Average Forecast:", round(np.mean(forecast), 4))
#         st.write("📈 Max Forecast:", round(np.max(forecast), 4))
#         st.write("📉 Min Forecast:", round(np.min(forecast), 4))

#         # =========================================================
#         # 📉 RESIDUAL DIAGNOSTICS (VERY IMPORTANT FOR MARKS)
#         # =========================================================

#         st.subheader("📉 Residual Diagnostics")

#         residuals = model.resid

#         fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

#         # -----------------------------
#         # 1. Residual Time Plot
#         # -----------------------------
#         axes[0].plot(residuals, color="blue")
#         axes[0].set_title("Residuals Over Time")
#         axes[0].axhline(0, linestyle="--", color="red")
#         axes[0].grid(True)

#         # -----------------------------
#         # 2. Histogram of Residuals
#         # -----------------------------
#         axes[1].hist(residuals.dropna(), bins=20, color="green", alpha=0.7)
#         axes[1].set_title("Residual Distribution")

#         # -----------------------------
#         # 3. ACF of Residuals
#         # -----------------------------
#         plot_acf(residuals.dropna(), ax=axes[2], lags=20)
#         axes[2].set_title("ACF of Residuals")

#         plt.tight_layout()
#         st.pyplot(fig2)

#         # =========================================================
#         # 📊 MODEL PERFORMANCE METRICS (OPTIONAL DISPLAY)
#         # =========================================================


#     except Exception as e:
#         st.error(f"SARIMA error: {e}")
import streamlit as st

st.set_page_config(page_title="Meteorite Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from statsmodels.graphics.tsaplots import plot_acf

# -----------------------------
# THEME
# -----------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b0f19, #05070d);
    color: #e6e6e6;
    font-family: 'Segoe UI', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

h1, h2, h3 {
    color: #4cc9f0;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #4cc9f0, #4361ee);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    border: none;
}

div.stButton > button:hover {
    transform: scale(1.02);
    transition: 0.2s;
}
.stTabs [data-baseweb="tab"] div {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #a1a1aa !important;
}

.stTabs [data-baseweb="tab"] {
    padding: 14px 28px !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #4cc9f0 !important;
}

.stTabs [aria-selected="true"] div {
    font-size: 20px !important;
    font-weight: 800 !important;
    color: #4cc9f0 !important;
}

div[data-testid="stPyplotChart"] {
    background: #0f172a;
    padding: 10px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

[data-testid="stMetric"] {
    background: #0f172a;
    padding: 10px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 30px;
    justify-content: center;
    padding: 10px 0px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    font-size: 18px;
    font-weight: 600;
    color: #a1a1aa;
    border-radius: 10px;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background-color: #111827;
    color: #4cc9f0 !important;
    box-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "backend", "data", "processed", "meteorite_final.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -----------------------------
# FIG SIZE
# -----------------------------
FIG_SIZE = (4.2, 2.6)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("""
<div style="
    text-align:center;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(90deg, #0f172a, #111827);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 10px;
">
    <h1 style="color:#4cc9f0; font-size:38px; margin-bottom:5px;">
        Meteorite Intelligence Dashboard
    </h1>
    <p style="color:#a1a1aa; font-size:16px;">
        Machine Learning • Forecasting • Space Data Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Prediction", "Forecasting"])

# =====================================================
# EDA
# =====================================================
with tab1:

    st.header("Dataset Overview")

    df_eda = df.copy()
    df_eda["year"] = pd.to_numeric(df_eda["year"], errors="coerce")
    df_eda["mass_tonnes"] = pd.to_numeric(df_eda["mass_tonnes"], errors="coerce")
    df_eda = df_eda.dropna(subset=["year", "mass_tonnes"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df_eda))
    col2.metric("Year Range", f"{int(df_eda.year.min())}-{int(df_eda.year.max())}")
    col3.metric("Avg Mass", round(df_eda.mass_tonnes.mean(), 4))
    col4.metric("Classes", df_eda["class"].nunique())

    yearly = df_eda.groupby("year").size()
    sample_df = df_eda.sample(min(3000, len(df_eda)))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mass Distribution")
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.hist(np.log10(df_eda["mass_tonnes"] + 1e-9), bins=35)
        st.pyplot(fig)

    with col2:
        st.subheader("Meteorite Count Trend")
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(yearly.index, yearly.values)
        st.pyplot(fig)

    st.subheader("Mass vs Year")
    fig, ax = plt.subplots(figsize=(3.8, 2.3))
    ax.scatter(sample_df["year"], np.log10(sample_df["mass_tonnes"] + 1e-9))
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    st.subheader("Top Classes")
    fig, ax = plt.subplots(figsize=(3.8, 2.3))
    df_eda["class"].value_counts().head(10).plot(kind="bar", ax=ax)
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    st.subheader("Map")
    map_df = df_eda.dropna(subset=["lat", "lon"])
    if len(map_df) > 0:
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}))

# =====================================================
# PREDICTION
# =====================================================
with tab2:

    st.header("Meteorite Class Prediction")

    try:
        @st.cache_resource
        def load_model():
            return joblib.load(os.path.join(BASE_DIR, "backend", "models", "class_model.pkl"))

        model = load_model()

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Year", 1800, 2025, 2000)
            mass = st.number_input("Mass (tonnes)", 0.000001, 10.0, 0.01)

        with col2:
            lat = st.number_input("Latitude", -90.0, 90.0, 20.0)
            lon = st.number_input("Longitude", -180.0, 180.0, 70.0)

        if st.button("Predict"):

            input_df = pd.DataFrame([{
                "year": year,
                "mass_tonnes": mass,
                "lat": lat,
                "lon": lon
            }])

            probs = model.predict_proba(input_df)[0]
            classes = model.classes_

            top_idx = np.argmax(probs)

            st.success(f"Predicted Class: {classes[top_idx]}")
            st.write(f"Confidence: {probs[top_idx]*100:.2f}%")

            top_3 = probs.argsort()[-3:][::-1]

            for i in top_3:
                st.write(f"{classes[i]} → {probs[i]*100:.2f}%")

            avg_mass = df[df["class"] == classes[top_idx]]["mass_tonnes"].mean()
            st.write(f"Average mass: {round(avg_mass,4)} tonnes")

            st.bar_chart(pd.DataFrame({
                "Class": classes[top_3],
                "Probability": probs[top_3]
            }).set_index("Class"))

    except Exception as e:
        st.error(f"Model error: {e}")

# =====================================================
# FORECASTING (FIXED PROPERLY)
# =====================================================
with tab3:

    st.header("Meteorite Forecasting (SARIMA)")

    try:
        # ---------------- LOAD FIRST ----------------
        model_arima = joblib.load(os.path.join(BASE_DIR, "backend", "models", "sarima_model.pkl"))
        series = joblib.load(os.path.join(BASE_DIR, "backend", "models", "year_series.pkl"))

        # ---------------- CLEAN SERIES ----------------
        series = series.dropna()
        series.index = series.index.astype(int)
        series = series[~series.index.duplicated()]
        series = series.sort_index()

        # optional filter
        series = series[series.index >= 1900]

        # ---------------- FORECAST ----------------
        years_ahead = st.slider("Forecast Years", 1, 20, 10)

        forecast_log = model_arima.forecast(steps=years_ahead)
        forecast = np.expm1(forecast_log)

        last_year = int(series.index.max())
        forecast_years = np.arange(last_year + 1, last_year + years_ahead + 1)

        # ---------------- PLOT ----------------
        fig, ax = plt.subplots(figsize=(6.5, 3.8))

        ax.plot(series.index, series.values,
                label="History", color="#4cc9f0", linewidth=2)

        ax.plot(forecast_years, forecast,
                marker="o", color="#f72585", linewidth=2, label="Forecast")

        ax.fill_between(forecast_years, forecast, alpha=0.15, color="#f72585")

        ax.set_xlim(1900, forecast_years[-1] + 1)

        ax.set_title("SARIMA Forecast - Meteorite Mass Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mass (tonnes)")
        ax.grid(alpha=0.25)
        ax.legend()

        st.pyplot(fig)

        # ---------------- METRICS ----------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Average", round(np.mean(forecast), 4))
        col2.metric("Max", round(np.max(forecast), 4))
        col3.metric("Min", round(np.min(forecast), 4))

        # ---------------- TABLE ----------------
        st.subheader("Forecast Values")

        forecast_df = pd.DataFrame({
            "Year": forecast_years,
            "Forecast Mass (tonnes)": np.round(forecast, 4)
        })

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # ---------------- RESIDUALS ----------------
        st.subheader("Residual Diagnostics")

        residuals = pd.Series(model_arima.resid).dropna()

        fig2, axes = plt.subplots(3, 1, figsize=(4.2, 5.2))

        axes[0].plot(residuals)
        axes[0].axhline(0, linestyle="--", color="red")
        axes[0].set_title("Residuals")

        axes[1].hist(residuals, bins=18, alpha=0.7)
        axes[1].set_title("Distribution")

        plot_acf(residuals, ax=axes[2], lags=15)
        axes[2].set_title("ACF")

        plt.tight_layout()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"SARIMA error: {e}")