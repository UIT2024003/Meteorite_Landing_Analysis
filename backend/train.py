import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ------------------------------
# SCRAPING (MINIMAL + STABLE)
# ------------------------------

print("Scraping data...")

import requests

url = "https://en.wikipedia.org/wiki/List_of_meteorite_falls"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers, verify=False)

tables = pd.read_html(response.text)

# 🔥 IMPORTANT: use correct table index
df = tables[2]

print("Scraping done!")

# ------------------------------
# CLEANING
# ------------------------------

df.columns = [str(c).lower() for c in df.columns]

rename_dict = {}
for c in df.columns:
    if 'name' in c:
        rename_dict[c] = 'name'
    elif 'year' in c:
        rename_dict[c] = 'year'
    elif 'mass' in c:
        rename_dict[c] = 'mass'
    elif 'class' in c:
        rename_dict[c] = 'recclass'

df = df.rename(columns=rename_dict)

df = df[['name', 'year', 'mass', 'recclass']]

df['mass'] = pd.to_numeric(df['mass'], errors='coerce') * 1000
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df.dropna(inplace=True)
df = df[df['year'] > 1900]

upper = df['mass'].quantile(0.99)
df = df[df['mass'] < upper]

df.to_csv("../data/meteorite_cleaned.csv", index=False)

print("Cleaning done:", df.shape)

# ------------------------------
# EDA (IMPROVED BUT SIMPLE)
# ------------------------------

print("Running EDA...")

# Mass distribution
plt.figure()
sns.histplot(df['mass'], bins=30, kde=True)
plt.title("Mass Distribution")
plt.savefig("../data/mass_distribution.png")

# Mass vs year
plt.figure()
sns.scatterplot(x=df['year'], y=df['mass'])
plt.title("Mass vs Year")
plt.savefig("../data/mass_vs_year.png")

# Top classes
plt.figure()
df['recclass'].value_counts().head(10).plot(kind='bar')
plt.title("Top Classes")
plt.savefig("../data/top_classes.png")

# Meteorite count per year
year_counts = df.groupby('year').size()
plt.figure()
plt.plot(year_counts.index, year_counts.values)
plt.title("Meteorite Count Over Time")
plt.savefig("../data/meteor_count.png")

# Correlation heatmap
plt.figure()
sns.heatmap(df[['year', 'mass']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("../data/correlation.png")

print("EDA saved!")

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------

df['log_mass'] = np.log1p(df['mass'])
df['year_scaled'] = df['year'] - df['year'].min()

encoder = LabelEncoder()
df['class_encoded'] = encoder.fit_transform(df['recclass'])

X = df[['year_scaled', 'class_encoded']]
y = df['log_mass']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# MODELS (SLIGHTLY IMPROVED)
# ------------------------------

models = {
    "Linear": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor()
}

results = {}

print("\nTraining models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = (mae, r2)

    print(f"{name} -> MAE: {mae:.4f}, R2: {r2:.4f}")

# ------------------------------
# SAVE BEST MODEL
# ------------------------------

best_model_name = max(results, key=lambda x: results[x][1])
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

joblib.dump(best_model, "../models/best_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(encoder, "../models/encoder.pkl")

print("Model saved!")

# ------------------------------
# MODEL COMPARISON
# ------------------------------

names = list(results.keys())
r2_scores = [results[n][1] for n in names]

plt.figure()
plt.bar(names, r2_scores)
plt.title("Model Comparison (R2)")
plt.savefig("../data/model_comparison.png")

print("Project completed 🚀")