# import pandas as pd
# import os
# import re


# # -----------------------------
# # MASS CONVERTER
# # -----------------------------
# def convert_mass_to_tonnes(x):
#     if pd.isna(x):
#         return None

#     x = str(x).lower().replace(",", "").strip()

#     match = re.search(r"\d+(\.\d+)?", x)
#     if not match:
#         return None

#     value = float(match.group())

#     if "kg" in x:
#         return value / 1000
#     elif "g" in x:
#         return value / 1_000_000
#     elif "tonne" in x or "t" in x:
#         return value

#     return None


# # -----------------------------
# # CLEAN SCRAPED DATA
# # -----------------------------
# def clean_scraped_data(path):
#     df = pd.read_csv(path)

#     print(f"\n📊 SCRAPED DATA SHAPE: {df.shape}")

#     # clean columns
#     df.columns = (
#         df.columns
#         .str.strip()
#         .str.replace("\xa0", " ", regex=False)
#         .str.replace("  ", " ", regex=False)
#         .str.lower()
#     )

#     # rename
#     df.rename(columns={
#         "meteorite name": "name",
#         "total mass in tonnes (1,000 kg)": "mass_tonnes",
#         "fall observation date": "date",
#         "classification": "class",
#         "country": "country",
#         "state, province, or region": "region",
#         "mass": "mass_text"
#     }, inplace=True)

#     # remove duplicate columns
#     df = df.loc[:, ~df.columns.duplicated()]

#     # mass handling
#     df["mass_tonnes"] = pd.to_numeric(df.get("mass_tonnes"), errors="coerce")

#     if "mass_text" in df.columns:
#         df["mass_from_text"] = df["mass_text"].apply(convert_mass_to_tonnes)
#         df["mass_tonnes"] = df["mass_tonnes"].fillna(df["mass_from_text"])

#     # date
#     if "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")

#     # year
#     df["year"] = df["date"].dt.year

#     return df


# # -----------------------------
# # CLEAN NASA CSV DATA
# # -----------------------------
# def clean_nasa_data(path):
#     df = pd.read_csv(path)

#     print(f"\n📊 NASA DATA SHAPE: {df.shape}")

#     # lower columns
#     df.columns = df.columns.str.lower().str.strip()

#     print("\n🔍 NASA Columns:", df.columns.tolist())

#     # rename to match schema
#     df.rename(columns={
#         "name": "name",
#         "recclass": "class",
#         "mass (g)": "mass_g",
#         "year": "date",
#         "reclat": "lat",
#         "reclong": "lon"
#     }, inplace=True)

#     # mass conversion (g → tonnes)
#     if "mass_g" in df.columns:
#         df["mass_tonnes"] = pd.to_numeric(df["mass_g"], errors="coerce") / 1_000_000

#     # date fix
#     if "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")

#     df["year"] = df["date"].dt.year

#     # add missing columns to match scraped schema
#     df["country"] = None
#     df["region"] = None

#     return df


# # -----------------------------
# # MAIN FUNCTION
# # -----------------------------
# def clean_data():
#     scraped_path = "data/raw/meteorite_raw.csv"
#     nasa_path = "data/raw/meteorite_landings_csv.csv"

#     scraped_df = clean_scraped_data(scraped_path)
#     nasa_df = clean_nasa_data(nasa_path)

#     # -----------------------------
#     # STANDARDIZE COLUMNS
#     # -----------------------------
#     final_cols = ["name", "mass_tonnes", "date", "year", "class", "country", "region"]

#     scraped_df = scraped_df[[col for col in final_cols if col in scraped_df.columns]]
#     nasa_df = nasa_df[[col for col in final_cols if col in nasa_df.columns]]

#     # -----------------------------
#     # MERGE DATA
#     # -----------------------------
#     final_df = pd.concat([scraped_df, nasa_df], ignore_index=True)

#     print(f"\n📊 TOTAL ROWS AFTER MERGE: {len(final_df)}")

#     # remove duplicates (important!)
#     final_df = final_df.drop_duplicates(subset=["name", "year"])

#     print(f"📊 AFTER REMOVING DUPLICATES: {len(final_df)}")

#     # light filtering (not aggressive)
#     final_df = final_df[
#         final_df["mass_tonnes"].notna() | final_df["date"].notna()
#     ]

#     print(f"📊 FINAL ROWS AFTER FILTER: {len(final_df)}")

#     # -----------------------------
#     # SAVE
#     # -----------------------------
#     os.makedirs("data/processed", exist_ok=True)

#     final_df.to_csv("data/processed/meteorite_final.csv", index=False)

#     print("\n🎉 FINAL CLEANED DATA READY!")
#     print(final_df.head())


# if __name__ == "__main__":
#     clean_data()
import pandas as pd
import os
import re


# -----------------------------
# MASS CONVERTER
# -----------------------------
def convert_mass_to_tonnes(x):
    if pd.isna(x):
        return None

    x = str(x).lower().replace(",", "").strip()

    match = re.search(r"\d+(\.\d+)?", x)
    if not match:
        return None

    value = float(match.group())

    if "kg" in x:
        return value / 1000
    elif "g" in x:
        return value / 1_000_000
    elif "tonne" in x or "t" in x:
        return value

    return None


# -----------------------------
# SAFE YEAR EXTRACTION
# -----------------------------
def extract_year_safe(date_col):
    return date_col.astype(str).str.extract(r"(\d{4})")[0].astype(float)


# -----------------------------
# CLEAN SCRAPED DATA
# -----------------------------
def clean_scraped_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.lower()
    )

    df.rename(columns={
        "meteorite name": "name",
        "total mass in tonnes (1,000 kg)": "mass_tonnes",
        "fall observation date": "date",
        "classification": "class",
        "state, province, or region": "region",
        "mass": "mass_text"
    }, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    # MASS
    df["mass_tonnes"] = pd.to_numeric(df.get("mass_tonnes"), errors="coerce")

    if "mass_text" in df.columns:
        df["mass_from_text"] = df["mass_text"].apply(convert_mass_to_tonnes)
        df["mass_tonnes"] = df["mass_tonnes"].fillna(df["mass_from_text"])

    # YEAR
    df["year"] = extract_year_safe(df.get("date"))

    # TEXT (SAFE FILL)
    df["country"] = df["country"] if "country" in df.columns else None
    df["region"] = df["region"] if "region" in df.columns else None
    df["class"] = df["class"].fillna("Unknown")

    return df


# -----------------------------
# CLEAN NASA DATA
# -----------------------------
def clean_nasa_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip()

    df.rename(columns={
        "recclass": "class",
        "mass (g)": "mass_g",
        "year": "date",
        "reclat": "lat",
        "reclong": "lon"
    }, inplace=True)

    # MASS
    df["mass_tonnes"] = pd.to_numeric(df.get("mass_g"), errors="coerce") / 1_000_000

    # YEAR
    df["year"] = extract_year_safe(df.get("date"))

    # KEEP LOCATION (IMPORTANT)
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

    # TEXT
    df["country"] = None
    df["region"] = None
    df["class"] = df["class"].fillna("Unknown")

    return df


# -----------------------------
# MAIN CLEAN FUNCTION
# -----------------------------
def clean_data():

    scraped_df = clean_scraped_data("data/raw/meteorite_raw.csv")
    nasa_df = clean_nasa_data("data/raw/meteorite_landings_csv.csv")

    # COMMON COLUMNS
    final_cols = [
        "name", "mass_tonnes", "year", "class",
        "country", "region", "lat", "lon"
    ]

    for col in final_cols:
        if col not in scraped_df.columns:
            scraped_df[col] = None
        if col not in nasa_df.columns:
            nasa_df[col] = None

    scraped_df = scraped_df[final_cols]
    nasa_df = nasa_df[final_cols]

    # MERGE
    df = pd.concat([scraped_df, nasa_df], ignore_index=True)

    print("\nAfter merge:", df.shape)

    # FILTERS
    df = df[
        (df["year"] > 1800) &
        (df["year"] < 2025)
    ]

    df = df[df["mass_tonnes"] > 1e-6]

    df = df[
        df["mass_tonnes"] < df["mass_tonnes"].quantile(0.995)
    ]

    # REMOVE DUPLICATES
    df = df.drop_duplicates(subset=["name", "year"])

    # FINAL CLEANING
    df["country"] = df["country"].fillna("Unknown")
    df["region"] = df["region"].fillna("Unknown")
    df["class"] = df["class"].fillna("Unknown")

    # SAVE
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/meteorite_final.csv", index=False)

    print("\n✅ FINAL DATA:", df.shape)
    print(df.head())


if __name__ == "__main__":
    clean_data()