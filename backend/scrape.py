# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# import os

# def scrape_data():
#     url = "https://datahub.io/core/meteorite-landings/r/meteorite-landings.csv"

#     headers = {"User-Agent": "Mozilla/5.0"}

#     try:
#         response = requests.get(url, headers=headers)

#         if response.status_code != 200:
#             print("❌ Failed:", response.status_code)
#             return

#         # This is still scraping (getting raw web data)
#         df = pd.read_csv(response.text.splitlines())

#         print("✅ Data scraped successfully!")
#         print("Shape:", df.shape)

#         os.makedirs("data/raw", exist_ok=True)
#         df.to_csv("data/raw/meteorite_raw.csv", index=False)

#         print("✅ Saved file")
#         print(df.head())

#     except Exception as e:
#         print("❌ Error:", e)


# if __name__ == "__main__":
#     scrape_data()


import pandas as pd
import requests
from io import StringIO
import os

def scrape_data():
    url = "https://en.wikipedia.org/wiki/List_of_meteorite_falls"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch data")
        return

    # Read all tables
    tables = pd.read_html(StringIO(response.text))

    print(f"Total tables found: {len(tables)}")

    # Collect all meteorite tables
    meteorite_tables = []

    for table in tables:
        if "Meteorite name" in table.columns:
            meteorite_tables.append(table)

    if not meteorite_tables:
        print("No meteorite tables found!")
        return

    # Combine all tables
    df = pd.concat(meteorite_tables, ignore_index=True)

    # Create folder
    os.makedirs("data/raw", exist_ok=True)

    # Save
    df.to_csv("data/raw/meteorite_raw.csv", index=False)

    print("✅ Full dataset scraped!")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    scrape_data()