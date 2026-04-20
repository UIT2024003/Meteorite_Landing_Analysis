import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv("data/processed/meteorite_final.csv")
    print("📊 Data Loaded:", df.shape)
    return df


# -----------------------------
# CLEAN DATA (IMPORTANT FIX)
# -----------------------------
def prepare(df):
    df = df.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mass_tonnes"] = pd.to_numeric(df["mass_tonnes"], errors="coerce")

    df = df.dropna(subset=["year", "mass_tonnes"])

    df["year"] = df["year"].astype(int)

    df = df[df["mass_tonnes"] > 0]

    # remove extreme outliers
    df = df[df["mass_tonnes"] < df["mass_tonnes"].quantile(0.99)]

    return df


# -----------------------------
# MAIN EDA
# -----------------------------
def run_eda(df):

    os.makedirs("data/plots", exist_ok=True)

    # =====================================================
    # 🔥 1. MASS DISTRIBUTION (FIXED PROPERLY)
    # =====================================================
    plt.figure(figsize=(7,5))

    # better transformation than log10(x+1)
    mass_log = np.log10(df["mass_tonnes"] + 1e-6)

    plt.hist(mass_log, bins=50, color="steelblue", edgecolor="black")

    plt.title("Meteorite Mass Distribution (Log10 Corrected)")
    plt.xlabel("Log10(Mass)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("data/plots/mass_distribution.png")
    plt.close()


    # =====================================================
    # 2. YEAR TREND
    # =====================================================
    yearly = df.groupby("year").size().rolling(3, min_periods=1).mean()

    plt.figure()
    plt.plot(yearly.index, yearly.values, color="orange")

    plt.title("Meteorite Trend Over Years")
    plt.xlabel("Year")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("data/plots/year_trend.png")
    plt.close()


    # =====================================================
    # 3. MASS VS YEAR (FIXED SCALE)
    # =====================================================
    plt.figure()

    plt.scatter(df["year"], np.log10(df["mass_tonnes"] + 1e-6), alpha=0.5)

    plt.title("Mass vs Year (Log10 Scale)")
    plt.xlabel("Year")
    plt.ylabel("Log10(Mass)")

    plt.tight_layout()
    plt.savefig("data/plots/mass_vs_year.png")
    plt.close()


    # =====================================================
    # 4. BUBBLE CHART (FIXED SCALING)
    # =====================================================
    bubble = df.groupby("year").agg({
        "mass_tonnes": "mean",
        "name": "count"
    }).reset_index()

    plt.figure(figsize=(7,5))

    plt.scatter(
        bubble["year"],
        bubble["mass_tonnes"],
        s=np.sqrt(bubble["name"]) * 25,
        alpha=0.6
    )

    plt.title("Bubble Chart (Year-wise Impact)")
    plt.xlabel("Year")
    plt.ylabel("Avg Mass")

    plt.tight_layout()
    plt.savefig("data/plots/bubble.png")
    plt.close()


    # =====================================================
    # 5. HEATMAP
    # =====================================================
    df_heat = df.copy()
    df_heat["class_code"] = df["class"].astype("category").cat.codes
    df_heat["country_code"] = df["country"].astype("category").cat.codes

    corr = df_heat[["year", "mass_tonnes", "class_code", "country_code"]].corr()

    plt.figure()

    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("data/plots/heatmap.png")
    plt.close()


    # =====================================================
    # 6. BULLET CHART (FIXED VISIBILITY)
    # =====================================================
    avg_mass = df["mass_tonnes"].mean()
    max_mass = df["mass_tonnes"].max()

    plt.figure(figsize=(6,2))

    # IMPORTANT FIX: show both clearly
    plt.barh(["Max"], [max_mass], color="lightgray")
    plt.barh(["Avg"], [avg_mass], color="steelblue")

    plt.title("Bullet Chart (Avg vs Max Mass)")
    plt.xlabel("Mass (tonnes)")

    plt.tight_layout()
    plt.savefig("data/plots/bullet.png")
    plt.close()


    # =====================================================
    # 7. MICRO TREND (FULLY FIXED + FALLBACK)
    # =====================================================
    plt.figure(figsize=(8,5))

    plotted = False

    top_regions = df["region"].dropna().value_counts().head(5).index

    for region in top_regions:
        sub = df[df["region"] == region].dropna(subset=["year"])

        trend = sub.groupby("year").size().sort_index()

        if len(trend) < 2:
            continue

        trend = trend.rolling(2, min_periods=1).mean()

        plt.plot(trend.index, trend.values, label=region)
        plotted = True

    # 🔥 FALLBACK (IMPORTANT FIX)
    if not plotted:
        top_countries = df["country"].dropna().value_counts().head(5).index

        for country in top_countries:
            sub = df[df["country"] == country]

            trend = sub.groupby("year").size().sort_index()

            if len(trend) < 2:
                continue

            plt.plot(trend.index, trend.values, label=country)

    if plotted or len(top_regions) > 0:
        plt.legend()

    plt.title("Micro Trend (Regions/Countries)")
    plt.xlabel("Year")
    plt.ylabel("Meteorite Count")

    plt.tight_layout()
    plt.savefig("data/plots/micro.png")
    plt.close()


    print("\n📊 EDA COMPLETED SUCCESSFULLY!")


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data()
    df = prepare(df)
    run_eda(df)


if __name__ == "__main__":
    main()