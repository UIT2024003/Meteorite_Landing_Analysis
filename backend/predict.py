import joblib
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = joblib.load("models/meteorite_model.pkl")
    year_min = joblib.load("models/year_min.pkl")
    return model, year_min


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(year):
    model, year_min = load_model()

    # scale year same way as training
    year_scaled = np.array([[year - year_min]])

    log_mass_pred = model.predict(year_scaled)

    # reverse log transform
    mass_pred = np.expm1(log_mass_pred)

    return mass_pred[0]


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    year = 2000

    result = predict(year)

    print(f"\n🔮 Predicted meteorite mass in {year}: {round(result, 4)} tonnes")