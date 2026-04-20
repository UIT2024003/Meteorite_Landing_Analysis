import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv("data/processed/meteorite_final.csv")

    df = df.dropna(subset=["mass_tonnes", "year", "class", "lat", "lon"])

    print("📊 Data Loaded:", df.shape)
    return df


# -----------------------------
# PIPELINE
# -----------------------------
def build_pipeline():

    numeric = ["year", "mass_tonnes", "lat", "lon"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric)
    ])

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# -----------------------------
# TRAIN MODEL (AutoML)
# -----------------------------
def train_model(df):

    X = df[["year", "mass_tonnes", "lat", "lon"]]
    y = df["class"]

    # ⚠️ reduce classes → keep top 10 only
    top_classes = y.value_counts().head(10).index
    df = df[df["class"].isin(top_classes)]

    X = df[["year", "mass_tonnes", "lat", "lon"]]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()

    param_dist = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=4,
        cv=2,
        verbose=1,
        n_jobs=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    preds = best_model.predict(X_test)

    print("\n📊 MODEL PERFORMANCE")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    return best_model


# -----------------------------
# SAVE
# -----------------------------
def save(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/class_model.pkl")
    print("✅ Model saved!")


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data()
    model = train_model(df)
    save(model)


if __name__ == "__main__":
    main()