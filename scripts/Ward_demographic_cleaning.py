"""
Prepare Pune Ward Demographic Data for PostGIS
-----------------------------------------------
Cleans pune_ward_reservation.csv (ward-level population data) for MapMyStore.
Per BE Report: population density is required for demand prediction and location suitability.

Data structure:
  - PMC (Pune Municipal Corporation): wards 1-41
  - PCMC (Pimpri-Chinchwad): wards 1-32
  - Empty rows separate the two bodies

Output: pune_wards_for_postgis.csv (saved to data/raw/)

Usage (from project root):
  python scripts/Ward_demographic_cleaning.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
INPUT_FILE = os.path.join(DATA_RAW, "pune_ward_reservation.csv")
OUTPUT_FILE = os.path.join(DATA_RAW, "pune_wards_for_postgis.csv")


def clean_ward_data():
    """Load, clean, and prepare ward demographic data for PostGIS."""
    print("=" * 70)
    print("WARD DEMOGRAPHIC DATA -> PostGIS")
    print("=" * 70)

    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, low_memory=False, encoding="cp1252")

    # Normalize column names (handle variations)
    df.columns = df.columns.str.strip()
    col_map = {
        "Ward": "ward_num",
        "Total_Population": "total_population",
        "Ward_Name": "ward_name",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Drop empty rows (separator between PMC and PCMC)
    df = df.dropna(subset=["ward_num", "total_population"], how="all")
    df = df[df["ward_num"].notna() & (df["ward_num"] != "")]

    # Parse numeric columns
    df["ward_num"] = pd.to_numeric(df["ward_num"], errors="coerce")
    df["total_population"] = pd.to_numeric(df["total_population"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows with missing required data
    df = df[
        df["total_population"].notna()
        & df["latitude"].notna()
        & df["longitude"].notna()
        & (df["total_population"] > 0)
    ].copy()

    # Assign zone (PMC vs PCMC) - empty rows separate them; first block is PMC, second is PCMC
    ward_nums = df["ward_num"].values
    zones = []
    current_zone = "PMC"
    prev_ward = 0
    for w in ward_nums:
        if w == 1 and prev_ward > 1:  # Ward restarts -> new zone
            current_zone = "PCMC"
        zones.append(current_zone)
        prev_ward = w
    df["zone"] = zones

    # Unique ward_id (PMC_1, PCMC_1, etc.)
    df["ward_id"] = df["zone"] + "_" + df["ward_num"].astype(int).astype(str)

    # Fill ward_name if missing
    df["ward_name"] = df["ward_name"].fillna("Unknown")

    # Clean special chars in ward names
    df["ward_name"] = df["ward_name"].astype(str).str.replace("�", "-", regex=False)

    # Add metadata
    df["data_source"] = "Pune Ward Reservation Data"
    df["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    # Reorder columns
    cols = [
        "ward_id",
        "zone",
        "ward_num",
        "ward_name",
        "latitude",
        "longitude",
        "total_population",
        "data_source",
        "last_updated",
    ]
    df = df[cols]

    return df


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)

    df = clean_ward_data()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Saved to {OUTPUT_FILE}")
    print(f"Total wards: {len(df)} (PMC: {(df['zone'] == 'PMC').sum()}, PCMC: {(df['zone'] == 'PCMC').sum()})")
    print(f"\nColumns: {list(df.columns)}")
    print("\nSample:")
    print(df.head(5).to_string(index=False))
    print("\n" + "=" * 70)
    print("Data ready for PostGIS. Run feature_pipeline.py to integrate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
