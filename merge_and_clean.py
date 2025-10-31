# merge_and_clean.py
import pandas as pd
import numpy as np
from pathlib import Path

# Directory and output
DATA_DIR = Path("dataset")
OUT_FILE = DATA_DIR / "merged_air_quality.csv"

# Only include CLEANED datasets (from your clean_datasets.py script)
files = [
    DATA_DIR / "cleaned_Location_data_2021.xlsx",
    DATA_DIR / "cleaned_Location_data_2022.xlsx",
    DATA_DIR / "cleaned_Location_data_2023.xlsx"
]

def load_and_standardize(path):
    """Load each cleaned Excel file and ensure consistent columns."""
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"⚠️ Error loading {path.name}: {e}")
        return pd.DataFrame()

    expected_cols = [
        "State / Union Territory",
        "City / town",
        "SO2 Annual Average",
        "NO2 Annual Average",
        "PM10 Annual Average",
        "PM2.5 Annual Average"
    ]

    # Add missing columns if any
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols]

    # Clean data
    df.replace(['NM', '-', ' ', '', 'na', 'NA', 'N/A'], np.nan, inplace=True)
    for col in expected_cols[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=["PM2.5 Annual Average"], inplace=True)
    return df

def main():
    all_data = []
    for f in files:
        if not f.exists():
            print(f"⚠️ {f.name} not found, skipping.")
            continue
        print(f"✅ Loading {f.name} ...")
        df = load_and_standardize(f)
        if not df.empty:
            print(f"   → {len(df)} rows loaded.")
            all_data.append(df)
        else:
            print(f"⚠️ No valid data found in {f.name}")

    if not all_data:
        print("❌ No valid datasets to merge.")
        return

    merged = pd.concat(all_data, ignore_index=True)
    merged.drop_duplicates(inplace=True)

    merged.to_csv(OUT_FILE, index=False)
    print("\n✅ Merged dataset saved successfully!")
    print("📄 File:", OUT_FILE)
    print("📊 Rows:", len(merged))
    print("📊 Columns:", merged.columns.tolist())

if __name__ == "__main__":
    main()
