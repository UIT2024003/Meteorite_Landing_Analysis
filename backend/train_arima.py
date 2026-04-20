import pandas as pd
import numpy as np
import os
import joblib
from pmdarima import auto_arima

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv("data/processed/meteorite_final.csv")

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df


# -----------------------------
# PREPARE TIME SERIES (FIXED)
# -----------------------------
def prepare_time_series(df):
    yearly = df.groupby("year").size().sort_index()

    # fill missing years
    full_years = np.arange(yearly.index.min(), yearly.index.max() + 1)
    yearly = yearly.reindex(full_years, fill_value=0)

    # 🔥 CRITICAL FIX: proper time index
    yearly.index = pd.date_range(
        start=str(yearly.index.min()),
        periods=len(yearly),
        freq="YS"
    )

    yearly = yearly.asfreq("YS")

    return yearly


# -----------------------------
# TRAIN ARIMA (FIXED)
# -----------------------------
def train_arima(series):

    model = auto_arima(
        series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action="ignore",
        max_p=3,
        max_q=3,
        max_d=2,
        stationary=False
    )

    print("\n📈 Best ARIMA Model:")
    print(model.summary())

    return model


# -----------------------------
# SAVE MODEL + SERIES (IMPORTANT FIX)
# -----------------------------
def save_model(model, series):
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/arima_model.pkl")
    joblib.dump(series, "models/year_series.pkl")

    print("\n💾 Model + Series saved successfully")


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data()
    series = prepare_time_series(df)

    model = train_arima(series)

    save_model(model, series)

    print("\n🎉 Training Complete!")


if __name__ == "__main__":
    main()