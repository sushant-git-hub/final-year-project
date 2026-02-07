"""
Prepare Pune Rental Data for PostGIS Storage
---------------------------------------------
Aggregates rental data to one record per locality with:
- Overall average rent per sqft
- Coordinates (latitude, longitude)
- Zone information
- Property count/confidence metrics

Output: Clean CSV ready for PostGIS import (saved to data/raw/)

Usage (from project root):
  python scripts/Rent_data_cleaning.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Project paths (run from project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

# Pune locality coordinates (major areas)
# These are approximate central coordinates for each locality
PUNE_COORDINATES = {
    # Central Pune
    'Shivajinagar': (18.5304, 73.8468),
    'Deccan': (18.5167, 73.8422),
    'Camp': (18.5089, 73.8800),
    'Koregaon Park': (18.5362, 73.8958),
    'Kothrud': (18.5074, 73.8077),
    'Sadashiv Peth': (18.5109, 73.8516),
    'Karve Nagar': (18.4881, 73.8226),
    
    # East Pune
    'Viman Nagar': (18.5679, 73.9143),
    'Kharadi': (18.5515, 73.9474),
    'Kalyani Nagar': (18.5465, 73.9062),
    'Magarpatta': (18.5156, 73.9295),
    'Hadapsar': (18.5089, 73.9260),
    'Mundhwa': (18.5362, 73.9318),
    'Yerawada': (18.5551, 73.8817),
    'Lohegaon': (18.5982, 73.9282),
    
    # West Pune
    'Baner': (18.5590, 73.7804),
    'Aundh': (18.5590, 73.8072),
    'Hinjewadi': (18.5912, 73.7337),
    'Balewadi': (18.5684, 73.7696),
    'Wakad': (18.6029, 73.7538),
    'Bavdhan': (18.5125, 73.7738),
    'Pashan': (18.5367, 73.7954),
    'Sus': (18.5598, 73.7562),
    
    # PCMC
    'Pimpri': (18.6298, 73.8009),
    'Chinchwad': (18.6479, 73.7972),
    'Akurdi': (18.6476, 73.7654),
    'Nigdi': (18.6543, 73.7702),
    'Pimple Saudagar': (18.5966, 73.8023),
    'Ravet': (18.6542, 73.7372),
    'Thergaon': (18.6100, 73.7900),
    
    # South Pune
    'Katraj': (18.4481, 73.8687),
    'Kondhwa': (18.4681, 73.8935),
    'Undri': (18.4626, 73.9159),
    'Wanowrie': (18.4864, 73.9095),
    'Bibvewadi': (18.4758, 73.8658),
    'Dhankawadi': (18.4626, 73.8509),
    'Narhe': (18.4605, 73.7956),
    'Warje': (18.4800, 73.8074),
}

def get_coordinates(locality):
    """Get lat/long for a locality"""
    
    locality_clean = str(locality).strip()
    
    # Direct match
    if locality_clean in PUNE_COORDINATES:
        return PUNE_COORDINATES[locality_clean]
    
    # Fuzzy match (case-insensitive, partial)
    locality_lower = locality_clean.lower()
    for key, coords in PUNE_COORDINATES.items():
        if key.lower() in locality_lower or locality_lower in key.lower():
            return coords
    
    # Default to Pune center if no match
    return (18.5204, 73.8567)  # Pune City Center

def categorize_zone(locality):
    """Categorize locality into zones"""
    locality_lower = str(locality).lower()
    
    central = ['shivajinagar', 'deccan', 'camp', 'koregaon', 'sadashiv', 'kothrud', 'karve', 'erandwane']
    east = ['viman nagar', 'kharadi', 'kalyani nagar', 'magarpatta', 'hadapsar', 'mundhwa', 'yerawada', 'lohegaon']
    west = ['baner', 'aundh', 'hinjewadi', 'balewadi', 'wakad', 'bavdhan', 'pashan', 'sus']
    pcmc = ['pimpri', 'chinchwad', 'akurdi', 'nigdi', 'pimple', 'ravet', 'thergaon']
    south = ['katraj', 'kondhwa', 'undri', 'wanowrie', 'bibvewadi', 'dhankawadi', 'narhe', 'warje']
    
    for keyword in central:
        if keyword in locality_lower:
            return 'Central Pune'
    for keyword in east:
        if keyword in locality_lower:
            return 'East Pune'
    for keyword in west:
        if keyword in locality_lower:
            return 'West Pune'
    for keyword in pcmc:
        if keyword in locality_lower:
            return 'PCMC'
    for keyword in south:
        if keyword in locality_lower:
            return 'South Pune'
    
    return 'Pune'

def aggregate_for_postgis():
    """Create aggregated dataset for PostGIS"""
    
    print("="*80)
    print("PREPARING DATA FOR POSTGIS")
    print("="*80)
    
    # Load the detailed real data
    df = pd.read_csv(os.path.join(DATA_RAW, "test.csv"))
    
    print(f"\nLoaded {len(df)} property records")
    
    # Extract locality from address
    df['locality'] = df['address'].apply(lambda x: str(x).split(',')[1].strip() if pd.notna(x) and ',' in str(x) else 'Unknown')
    
    # Filter valid records (has area and rent)
    df_valid = df[(df['area'] > 0) & (df['rent'] > 0)].copy()
    
    print(f"Valid records with area & rent: {len(df_valid)}")
    
    # Calculate rent per sqft
    df_valid['rent_per_sqft'] = df_valid['rent'] / df_valid['area']
    
    # Remove extreme outliers
    df_valid = df_valid[
        (df_valid['rent_per_sqft'] >= 5) & 
        (df_valid['rent_per_sqft'] <= 200)
    ]
    
    print(f"After removing outliers: {len(df_valid)}")
    
    # Group by locality and aggregate
    print("\nAggregating by locality...")
    
    locality_data = df_valid.groupby('locality').agg({
        'rent': ['mean', 'median', 'min', 'max', 'std'],
        'rent_per_sqft': ['mean', 'median', 'std'],
        'area': ['mean', 'median'],
        'bedroom': 'mean',
        'bathrooms': 'mean',
        'locality': 'count'
    }).reset_index()
    
    # Flatten column names
    locality_data.columns = [
        'locality',
        'avg_rent',
        'median_rent', 
        'min_rent',
        'max_rent',
        'std_rent',
        'avg_rent_per_sqft',
        'median_rent_per_sqft',
        'std_rent_per_sqft',
        'avg_area_sqft',
        'median_area_sqft',
        'avg_bedrooms',
        'avg_bathrooms',
        'sample_size'
    ]
    
    # Add zone classification
    locality_data['zone'] = locality_data['locality'].apply(categorize_zone)
    
    # Add coordinates
    coords = locality_data['locality'].apply(get_coordinates)
    locality_data['latitude'] = coords.apply(lambda x: x[0])
    locality_data['longitude'] = coords.apply(lambda x: x[1])
    
    # Estimate commercial rent (15% premium over residential)
    locality_data['commercial_rent_per_sqft'] = (locality_data['avg_rent_per_sqft'] * 1.15).round(2)
    
    # Calculate confidence score based on sample size
    locality_data['confidence'] = locality_data['sample_size'].apply(
        lambda x: 'High' if x >= 10 else ('Medium' if x >= 5 else 'Low')
    )
    
    # Add tier classification based on rent
    def classify_tier(rent_per_sqft):
        if rent_per_sqft >= 40:
            return 'Premium'
        elif rent_per_sqft >= 25:
            return 'Grade A'
        elif rent_per_sqft >= 15:
            return 'Grade B'
        else:
            return 'Grade C'
    
    locality_data['tier'] = locality_data['avg_rent_per_sqft'].apply(classify_tier)
    
    # Round numeric columns
    numeric_cols = [
        'avg_rent', 'median_rent', 'min_rent', 'max_rent', 'std_rent',
        'avg_rent_per_sqft', 'median_rent_per_sqft', 'std_rent_per_sqft',
        'commercial_rent_per_sqft', 'avg_area_sqft', 'median_area_sqft'
    ]
    
    for col in numeric_cols:
        locality_data[col] = locality_data[col].round(2)
    
    locality_data['avg_bedrooms'] = locality_data['avg_bedrooms'].round(1)
    locality_data['avg_bathrooms'] = locality_data['avg_bathrooms'].round(1)
    
    # Add metadata
    locality_data['data_source'] = 'Real Property Listings'
    locality_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    
    # Reorder columns for PostGIS
    final_columns = [
        'locality',
        'zone',
        'tier',
        'latitude',
        'longitude',
        'avg_rent_per_sqft',
        'median_rent_per_sqft',
        'commercial_rent_per_sqft',
        'avg_rent',
        'median_rent',
        'min_rent',
        'max_rent',
        'avg_area_sqft',
        'median_area_sqft',
        'sample_size',
        'confidence',
        'avg_bedrooms',
        'avg_bathrooms',
        'data_source',
        'last_updated'
    ]
    
    locality_data = locality_data[final_columns]
    
    # Sort by zone and then by locality name
    locality_data = locality_data.sort_values(['zone', 'locality'])
    
    print(f"\n✓ Aggregated to {len(locality_data)} unique localities")
    
    return locality_data

def create_postgis_ready_csv():
    """Create final CSV for PostGIS import"""
    
    # Aggregate data
    locality_df = aggregate_for_postgis()
    
    # Save to CSV
    output_file = os.path.join(DATA_RAW, "pune_localities_for_postgis.csv")
    locality_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("✓ SUCCESS - PostGIS Ready Data Created!")
    print("="*80)
    
    print(f"\nOutput File: {output_file}")
    print(f"Total Localities: {len(locality_df)}")
    print(f"\nColumns (20 total):")
    for i, col in enumerate(locality_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n" + "="*80)
    print("DATA SUMMARY BY ZONE:")
    print("="*80)
    
    zone_summary = locality_df.groupby('zone').agg({
        'locality': 'count',
        'avg_rent_per_sqft': 'mean',
        'commercial_rent_per_sqft': 'mean',
        'sample_size': 'sum'
    }).round(2)
    
    zone_summary.columns = ['Localities', 'Avg Residential Rent/sqft', 'Avg Commercial Rent/sqft', 'Total Properties']
    print(zone_summary.to_string())
    
    print("\n" + "="*80)
    print("DATA SUMMARY BY TIER:")
    print("="*80)
    
    tier_summary = locality_df.groupby('tier').agg({
        'locality': 'count',
        'avg_rent_per_sqft': 'mean',
        'commercial_rent_per_sqft': 'mean'
    }).round(2)
    
    tier_summary.columns = ['Localities', 'Avg Residential Rent/sqft', 'Avg Commercial Rent/sqft']
    print(tier_summary.to_string())
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First 10 Localities):")
    print("="*80)
    
    display_cols = ['locality', 'zone', 'tier', 'latitude', 'longitude', 
                    'avg_rent_per_sqft', 'commercial_rent_per_sqft', 'sample_size', 'confidence']
    
    print(locality_df[display_cols].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 10 MOST EXPENSIVE LOCALITIES:")
    print("="*80)
    
    top_10 = locality_df.nlargest(10, 'commercial_rent_per_sqft')[
        ['locality', 'zone', 'commercial_rent_per_sqft', 'sample_size']
    ]
    
    for i, row in top_10.iterrows():
        print(f"  {row['locality']:30s} {row['zone']:15s} ₹{row['commercial_rent_per_sqft']:6.2f}/sqft ({int(row['sample_size'])} properties)")
    
    print("\n" + "="*80)
    print("TOP 10 BUDGET-FRIENDLY LOCALITIES:")
    print("="*80)
    
    budget_10 = locality_df.nsmallest(10, 'commercial_rent_per_sqft')[
        ['locality', 'zone', 'commercial_rent_per_sqft', 'sample_size']
    ]
    
    for i, row in budget_10.iterrows():
        print(f"  {row['locality']:30s} {row['zone']:15s} ₹{row['commercial_rent_per_sqft']:6.2f}/sqft ({int(row['sample_size'])} properties)")
    
    print("\n" + "="*80)
    print("POSTGIS IMPORT INSTRUCTIONS:")
    print("="*80)
    
    print("""
1. CREATE TABLE in PostGIS:
   
   CREATE TABLE pune_localities (
       id SERIAL PRIMARY KEY,
       locality VARCHAR(255) NOT NULL,
       zone VARCHAR(50),
       tier VARCHAR(20),
       latitude DECIMAL(10, 6),
       longitude DECIMAL(10, 6),
       avg_rent_per_sqft DECIMAL(10, 2),
       median_rent_per_sqft DECIMAL(10, 2),
       commercial_rent_per_sqft DECIMAL(10, 2),
       avg_rent DECIMAL(10, 2),
       median_rent DECIMAL(10, 2),
       min_rent DECIMAL(10, 2),
       max_rent DECIMAL(10, 2),
       avg_area_sqft DECIMAL(10, 2),
       median_area_sqft DECIMAL(10, 2),
       sample_size INTEGER,
       confidence VARCHAR(20),
       avg_bedrooms DECIMAL(3, 1),
       avg_bathrooms DECIMAL(3, 1),
       data_source VARCHAR(255),
       last_updated DATE,
       geom GEOMETRY(Point, 4326)
   );

2. IMPORT CSV:
   
   COPY pune_localities(
       locality, zone, tier, latitude, longitude,
       avg_rent_per_sqft, median_rent_per_sqft, commercial_rent_per_sqft,
       avg_rent, median_rent, min_rent, max_rent,
       avg_area_sqft, median_area_sqft, sample_size, confidence,
       avg_bedrooms, avg_bathrooms, data_source, last_updated
   )
   FROM '/path/to/pune_localities_for_postgis.csv'
   DELIMITER ','
   CSV HEADER;

3. CREATE GEOMETRY COLUMN:
   
   UPDATE pune_localities
   SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);

4. CREATE SPATIAL INDEX:
   
   CREATE INDEX pune_localities_geom_idx 
   ON pune_localities 
   USING GIST(geom);

5. VERIFY DATA:
   
   SELECT locality, zone, 
          ST_AsText(geom) as location,
          commercial_rent_per_sqft
   FROM pune_localities
   LIMIT 10;
    """)
    
    print("="*80)
    print("✓ Data ready for PostGIS import!")
    print("="*80)
    
    return locality_df

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("PUNE RENTAL DATA -> PostGIS")
    print("="*80)
    print("\nProcessing real property data for PostGIS storage...")
    print("This will create ONE record per locality with aggregated data.\n")
    
    # Create PostGIS-ready dataset
    df = create_postgis_ready_csv()
    
    print("\n✓ Complete! Use 'pune_localities_for_postgis.csv' for PostGIS import")
    
    return df

if __name__ == "__main__":
    main()
