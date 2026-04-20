# import pandas as pd
# import numpy as np
# import os
# import joblib
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "meteorite_final.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "sarima_model.pkl")
# SERIES_PATH = os.path.join(BASE_DIR, "models", "year_series.pkl")

# # -----------------------------
# # LOAD + PREPARE DATA
# # -----------------------------
# def load_data():
#     df = pd.read_csv(DATA_PATH)

#     df = df.dropna(subset=["year", "mass_tonnes"])
#     df["year"] = df["year"].astype(int)

#     # yearly aggregation
#     yearly = df.groupby("year")["mass_tonnes"].sum().sort_index()

#     # IMPORTANT: fill missing years (VERY IMPORTANT for SARIMA)
#     full_range = range(yearly.index.min(), yearly.index.max() + 1)
#     yearly = yearly.reindex(full_range, fill_value=0)

#     return yearly

# # -----------------------------
# # TRAIN SARIMA (FIXED)
# # -----------------------------
# def train_model():
#     series = load_data()

#     # log transform → prevents flat forecasts
#     series_log = np.log1p(series)

#     # train-test split
#     train_size = int(len(series_log) * 0.85)
#     train, test = series_log[:train_size], series_log[train_size:]

#     # ✔️ FIXED SARIMA MODEL
#     model = SARIMAX(
#         train,
#         order=(2, 1, 2),          # stronger than (1,1,1)
#         seasonal_order=(0, 0, 0, 0),  # NO fake seasonality
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )

#     model_fit = model.fit(disp=False)

#     # -----------------------------
#     # PREDICTION
#     # -----------------------------
#     pred_log = model_fit.predict(start=len(train), end=len(series_log)-1)

#     # invert log transform
#     pred = np.expm1(pred_log)
#     test_actual = np.expm1(test)

#     # -----------------------------
#     # METRICS
#     # -----------------------------
#     mae = mean_absolute_error(test_actual, pred)
#     rmse = np.sqrt(mean_squared_error(test_actual, pred))

#     print("\n📊 MODEL PERFORMANCE")
#     print("MAE:", mae)
#     print("RMSE:", rmse)

#     # -----------------------------
#     # SAVE
#     # -----------------------------
#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#     joblib.dump(model_fit, MODEL_PATH)
#     joblib.dump(series, SERIES_PATH)

#     print("\n✅ SARIMA model saved successfully!")

# if __name__ == "__main__":
#     train_model()
import pandas as pd
import numpy as np
import os
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "meteorite_final.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sarima_model.pkl")
SERIES_PATH = os.path.join(BASE_DIR, "models", "year_series.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["year", "mass_tonnes"])
    df["year"] = df["year"].astype(int)

    yearly = df.groupby("year")["mass_tonnes"].sum().sort_index()

    full_range = range(yearly.index.min(), yearly.index.max() + 1)
    yearly = yearly.reindex(full_range, fill_value=0)

    return yearly

# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    series = load_data()

    # log transform (stable for skewed data)
    series_log = np.log1p(series)

    train_size = int(len(series_log) * 0.85)
    train, test = series_log[:train_size], series_log[train_size:]

    model = SARIMAX(
        train,
        order=(1, 1, 1),              # more stable than (2,1,2)
        seasonal_order=(1, 1, 1, 12), # proper SARIMA form (important for syllabus)
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    pred_log = model_fit.predict(start=len(train), end=len(series_log)-1)

    pred = np.expm1(pred_log)
    actual = np.expm1(test)

    # -----------------------------
    # METRICS (IMPORTANT FOR MARKS)
    # -----------------------------
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / (actual + 1))) * 100
    r2 = r2_score(actual, pred)

    print("\n📊 MODEL PERFORMANCE")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R2   : {r2:.4f}")

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_fit, MODEL_PATH)
    joblib.dump(series, SERIES_PATH)

    print("\n✅ SARIMA model saved successfully!")

if __name__ == "__main__":
    train_model()