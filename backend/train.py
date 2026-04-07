# ------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import requests
import warnings
import joblib

warnings.filterwarnings("ignore")

# ------------------------------
# 2. WEB SCRAPING
# ------------------------------
print("Scraping data from Wikipedia...")

url = "https://en.wikipedia.org/wiki/List_of_meteorite_falls"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)

df = tables[0]   # SAME AS YOUR ORIGINAL

print("Scraping completed!")
print("Columns:", df.columns)

# ------------------------------
# 3. DATA CLEANING (FIXED ONLY)
# ------------------------------
print("\nStarting Data Cleaning...")

df.columns = [str(c).lower() for c in df.columns]

# 🔥 FIX: extract year from date column
if 'fall observation  date' in df.columns:
    df['year'] = pd.to_datetime(df['fall observation  date'], errors='coerce').dt.year

# rename columns
rename_dict = {}
for c in df.columns:
    if 'name' in c:
        rename_dict[c] = 'name'
    elif 'mass' in c:
        rename_dict[c] = 'mass'
    elif 'class' in c:
        rename_dict[c] = 'recclass'

df = df.rename(columns=rename_dict)

# keep required columns
df = df[['name', 'year', 'mass', 'recclass']]

# convert types
df['mass'] = pd.to_numeric(df['mass'], errors='coerce') * 1000
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df = df.dropna()
df = df[df['year'] > 1900]

upper = df['mass'].quantile(0.99)
df = df[df['mass'] < upper]

df.to_csv("../data/meteorite_cleaned.csv", index=False)

print("Cleaning Completed!")
print("Shape:", df.shape)

# ------------------------------
# 4. EDA (SLIGHTLY IMPROVED)
# ------------------------------
print("\nPerforming EDA...")

# Mass distribution
plt.figure()
sns.histplot(df['mass'], bins=30, kde=True)
plt.title("Mass Distribution")
plt.savefig("../data/mass_distribution.png")

# Meteorites per year
year_counts = df.groupby('year').size()
plt.figure()
plt.plot(year_counts.index, year_counts.values)
plt.title("Meteorites per Year")
plt.savefig("../data/year_count.png")

# Mass vs year
plt.figure()
sns.scatterplot(x=df['year'], y=df['mass'])
plt.title("Mass vs Year")
plt.savefig("../data/mass_vs_year.png")

print("EDA Completed!")

# ------------------------------
# 5. REGRESSION (IMPROVED SLIGHTLY)
# ------------------------------
print("\nTraining Regression Model...")

df['log_mass'] = np.log1p(df['mass'])
df['year_scaled'] = df['year'] - df['year'].min()

X = df[['year_scaled']]
y = df['log_mass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 4))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

joblib.dump(model, "../models/meteorite_model.pkl")
joblib.dump(df['year'].min(), "../models/year_min.pkl")

print("✅ Model saved!")

# ------------------------------
# 6. TIME SERIES (UNCHANGED)
# ------------------------------
print("\nForecasting Meteorite Counts...")

year_data = df.groupby('year').size().rolling(3, min_periods=1).mean()

model_arima = ARIMA(year_data, order=(1,1,1))
model_fit = model_arima.fit()

forecast = model_fit.forecast(steps=5)

future_years = range(int(year_data.index.max())+1,
                     int(year_data.index.max())+6)

for y_val, count in zip(future_years, forecast):
    print(f"{y_val}: {int(count)}")

print("\n✅ Project Completed Successfully!")