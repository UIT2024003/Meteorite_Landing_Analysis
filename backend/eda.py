# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt


# # -----------------------------
# # LOAD DATA
# # -----------------------------
# def load_data():
#     df = pd.read_csv("data/processed/meteorite_final.csv")
#     print("📊 Data Loaded:", df.shape)
#     return df


# # -----------------------------
# # CLEAN DATA
# # -----------------------------
# def prepare(df):
#     df = df.copy()

#     # convert types
#     df["year"] = pd.to_numeric(df["year"], errors="coerce")
#     df["mass_tonnes"] = pd.to_numeric(df["mass_tonnes"], errors="coerce")

#     # drop missing
#     df = df.dropna(subset=["year", "mass_tonnes"])

#     # fix types
#     df["year"] = df["year"].astype(int)

#     # remove invalid values
#     df = df[df["mass_tonnes"] > 0]

#     # remove fake/default years
#     df = df[df["year"] > 1800]

#     # remove extreme outliers
#     df = df[df["mass_tonnes"] < df["mass_tonnes"].quantile(0.99)]

#     return df


# # -----------------------------
# # MAIN EDA
# # -----------------------------
# def run_eda(df):

#     os.makedirs("data/plots", exist_ok=True)

#     # =====================================================
#     # 1. MASS DISTRIBUTION (FIXED)
#     # =====================================================
#     plt.figure(figsize=(7,5))

#     mass = df["mass_tonnes"]

#     # remove extremely tiny noise
#     mass = mass[mass > mass.quantile(0.01)]

#     mass_log = np.log10(mass)

#     plt.hist(mass_log, bins=40)

#     plt.title("Meteorite Mass Distribution (Log Scale)")
#     plt.xlabel("Log10(Mass)")
#     plt.ylabel("Frequency")

#     plt.tight_layout()
#     plt.savefig("data/plots/mass_distribution.png")
#     plt.close()


#     # =====================================================
#     # 2. YEAR TREND (CLEANED)
#     # =====================================================
#     yearly = df.groupby("year").size()

#     yearly = yearly.rolling(3, min_periods=1).mean()

#     plt.figure()

#     plt.plot(yearly.index, yearly.values)

#     plt.title("Meteorite Trend Over Years")
#     plt.xlabel("Year")
#     plt.ylabel("Count")

#     plt.tight_layout()
#     plt.savefig("data/plots/year_trend.png")
#     plt.close()


#     # =====================================================
#     # 3. MASS VS YEAR
#     # =====================================================
#     plt.figure()

#     plt.scatter(df["year"], np.log10(df["mass_tonnes"]), alpha=0.5)

#     plt.title("Mass vs Year (Log Scale)")
#     plt.xlabel("Year")
#     plt.ylabel("Log10(Mass)")

#     plt.tight_layout()
#     plt.savefig("data/plots/mass_vs_year.png")
#     plt.close()


#     # =====================================================
#     # 4. BUBBLE CHART
#     # =====================================================
#     bubble = df.groupby("year").agg({
#         "mass_tonnes": "mean",
#         "name": "count"
#     }).reset_index()

#     plt.figure(figsize=(7,5))

#     plt.scatter(
#         bubble["year"],
#         bubble["mass_tonnes"],
#         s=np.sqrt(bubble["name"]) * 25,
#         alpha=0.6
#     )

#     plt.title("Bubble Chart (Year-wise Impact)")
#     plt.xlabel("Year")
#     plt.ylabel("Average Mass")

#     plt.tight_layout()
#     plt.savefig("data/plots/bubble.png")
#     plt.close()


#     # =====================================================
#     # 5. HEATMAP
#     # =====================================================
#     df_heat = df.copy()

#     df_heat["class_code"] = df_heat["class"].astype("category").cat.codes
#     df_heat["country_code"] = df_heat["country"].astype("category").cat.codes

#     corr = df_heat[["year", "mass_tonnes", "class_code", "country_code"]].corr()

#     plt.figure()

#     plt.imshow(corr, cmap="coolwarm")
#     plt.colorbar()

#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
#     plt.yticks(range(len(corr.columns)), corr.columns)

#     plt.title("Correlation Heatmap")

#     plt.tight_layout()
#     plt.savefig("data/plots/heatmap.png")
#     plt.close()


#     # =====================================================
#     # 6. BULLET CHART
#     # =====================================================
#     avg_mass = df["mass_tonnes"].mean()
#     max_mass = df["mass_tonnes"].max()

#     plt.figure(figsize=(6,2))

#     plt.barh(["Max"], [max_mass])
#     plt.barh(["Avg"], [avg_mass])

#     plt.title("Avg vs Max Mass")
#     plt.xlabel("Mass (tonnes)")

#     plt.tight_layout()
#     plt.savefig("data/plots/bullet.png")
#     plt.close()


#     # =====================================================
#     # 7. MICRO TREND (FIXED USING CLASS)
#     # =====================================================
#     plt.figure(figsize=(8,5))

#     top_classes = df["class"].dropna().value_counts().head(5).index

#     for cls in top_classes:
#         sub = df[df["class"] == cls]

#         trend = sub.groupby("year").size().sort_index()

#         if len(trend) < 2:
#             continue

#         trend = trend.rolling(3, min_periods=1).mean()

#         plt.plot(trend.index, trend.values, label=cls)

#     plt.legend()
#     plt.title("Micro Trend (Top Meteorite Classes)")
#     plt.xlabel("Year")
#     plt.ylabel("Count")

#     plt.tight_layout()
#     plt.savefig("data/plots/micro.png")
#     plt.close()


#     # =====================================================
#     # 8. TOP CLASSES (NEW INSIGHT)
#     # =====================================================
#     top_classes = df["class"].value_counts().head(10)

#     plt.figure(figsize=(8,5))

#     top_classes.plot(kind="bar")

#     plt.title("Top 10 Meteorite Classes")
#     plt.xlabel("Class")
#     plt.ylabel("Count")

#     plt.tight_layout()
#     plt.savefig("data/plots/top_classes.png")
#     plt.close()


#     print("\n✅ EDA COMPLETED SUCCESSFULLY!")


# # -----------------------------
# # MAIN
# # -----------------------------
# def main():
#     df = load_data()
#     df = prepare(df)
#     run_eda(df)


# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    path = "data/processed/meteorite_final.csv"
    print("📂 Loading from:", os.path.abspath(path))

    if not os.path.exists(path):
        print("❌ File NOT FOUND")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print("📊 Raw data shape:", df.shape)
    return df


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare(df):
    df = df.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mass_tonnes"] = pd.to_numeric(df["mass_tonnes"], errors="coerce")

    df = df.dropna(subset=["year", "mass_tonnes"])
    df["year"] = df["year"].astype(int)

    # basic cleaning
    df = df[df["mass_tonnes"] > 0]
    df = df[df["year"] > 1800]

    # remove extreme outliers
    df = df[df["mass_tonnes"] < df["mass_tonnes"].quantile(0.99)]

    return df


# -----------------------------
# RUN EDA
# -----------------------------
def run_eda(df):

    os.makedirs("data/plots", exist_ok=True)

    df = df.copy()

    # -----------------------------
    # SAFE CLEANING
    # -----------------------------
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mass_tonnes"] = pd.to_numeric(df["mass_tonnes"], errors="coerce")

    # SAFE LAT/LON
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    else:
        df["lat"] = np.nan

    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    else:
        df["lon"] = np.nan

    df = df.dropna(subset=["year", "mass_tonnes"])
    df = df[(df["year"] > 1800) & (df["year"] < 2025)]
    df = df[df["mass_tonnes"] > 0]

    print("🧹 After cleaning:", df.shape)

    if df.empty:
        print("❌ No data available after cleaning")
        return

    # =====================================================
    # 1. MASS DISTRIBUTION
    # =====================================================
    plt.figure()

    mass_log = np.log10(df["mass_tonnes"] + 1e-9)

    plt.hist(mass_log, bins=50)

    plt.title("Mass Distribution (Log10)")
    plt.xlabel("Log10(Mass Tonnes)")
    plt.ylabel("Frequency")

    plt.savefig("data/plots/mass.png")
    plt.close()

    # =====================================================
    # 2. YEAR TREND
    # =====================================================
    yearly = df.groupby("year").size()

    plt.figure()
    plt.plot(yearly.index, yearly.values)

    plt.title("Meteorite Count Over Years")
    plt.xlabel("Year")
    plt.ylabel("Count")

    plt.savefig("data/plots/year.png")
    plt.close()

    # =====================================================
    # 3. MASS VS YEAR
    # =====================================================
    plt.figure()

    sample_df = df.sample(min(5000, len(df)))

    plt.scatter(
        sample_df["year"],
        np.log10(sample_df["mass_tonnes"] + 1e-9),
        alpha=0.4
    )

    plt.title("Mass vs Year")
    plt.xlabel("Year")
    plt.ylabel("Log10(Mass)")

    plt.savefig("data/plots/mass_year.png")
    plt.close()

    # =====================================================
    # 4. TOP CLASSES
    # =====================================================
    plt.figure()

    df["class"].fillna("Unknown").value_counts().head(10).plot(kind="bar")

    plt.title("Top Meteorite Classes")

    plt.savefig("data/plots/classes.png")
    plt.close()

    # =====================================================
    # 5. CLASS MASS (MEDIAN)
    # =====================================================
    plt.figure()

    df.groupby("class")["mass_tonnes"].median()\
        .sort_values(ascending=False)\
        .head(10)\
        .plot(kind="bar")

    plt.title("Median Mass by Class")

    plt.savefig("data/plots/class_mass.png")
    plt.close()

    # =====================================================
    # 6. GEO MAP
    # =====================================================
    geo_df = df.dropna(subset=["lat", "lon"])

    print("🌍 Geo data:", geo_df.shape)

    if len(geo_df) > 0:
        plt.figure()

        geo_sample = geo_df.sample(min(5000, len(geo_df)))

        plt.scatter(
            geo_sample["lon"],
            geo_sample["lat"],
            alpha=0.4
        )

        plt.title("Meteorite Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.savefig("data/plots/map.png")
        plt.close()
    else:
        print("⚠️ No geo data available")

    print("✅ EDA COMPLETED SUCCESSFULLY")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting EDA...")

    df = load_data()

    if df.empty:
        print("❌ Dataset is empty or not found")
    else:
        df = prepare(df)
        print("🧹 After prepare:", df.shape)

        if df.empty:
            print("❌ No data after prepare step")
        else:
            run_eda(df)