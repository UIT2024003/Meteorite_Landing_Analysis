import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv("data/processed/meteorite_final.csv")

    df = df.dropna(subset=["mass_tonnes", "year", "country", "region", "class"])

    print("📊 Data Loaded:", df.shape)
    return df


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(df):

    # fill missing categorical
    df["country"] = df["country"].fillna("Unknown")
    df["region"] = df["region"].fillna("Unknown")
    df["class"] = df["class"].fillna("Unknown")

    # encoders
    encoders = {}

    for col in ["country", "region", "class"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[["year", "country", "region", "class"]]
    y = df["mass_tonnes"]

    return X, y, encoders


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n📈 MODEL PERFORMANCE")
    print("MAE:", round(mean_absolute_error(y_test, preds), 4))
    print("R2 Score:", round(r2_score(y_test, preds), 4))

    return model


# -----------------------------
# SAVE MODEL
# -----------------------------
def save(model, encoders):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/mass_model.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    print("\n✅ Model saved successfully!")


# -----------------------------
# MAIN
# -----------------------------
def main():

    df = load_data()
    X, y, encoders = preprocess(df)
    model = train_model(X, y)
    save(model, encoders)

    print("\n🎉 TRAINING COMPLETE")


if __name__ == "__main__":
    main()