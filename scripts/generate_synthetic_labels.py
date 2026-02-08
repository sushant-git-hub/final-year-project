"""
Generate Synthetic Training Labels for ML Model
Creates realistic revenue/success labels based on existing features
"""

import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def connect_to_db():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def fetch_grid_features(conn):
    """Fetch all grid cell features from database"""
    query = """
    SELECT 
        g.cell_id,
        g.center_lat,
        g.center_lon,
        COALESCE(p.competitor_count, 0) as competitor_count,
        COALESCE(p.nearest_m, 9999) as nearest_competitor_m,
        COALESCE(r.road_density_km, 0) as road_density_km,
        COALESCE(r.major_dist_m, 9999) as major_road_dist_m,
        COALESCE(rt.commercial_rent_per_sqft, 50) as rent_per_sqft,
        COALESCE(d.total_population, 0) as population
    FROM grid_cells g
    LEFT JOIN poi_features p ON g.cell_id = p.cell_id
    LEFT JOIN road_features r ON g.cell_id = r.cell_id
    LEFT JOIN rental_features rt ON g.cell_id = rt.cell_id
    LEFT JOIN demographic_features d ON g.cell_id = d.cell_id
    """
    
    df = pd.read_sql(query, conn)
    return df


def calculate_location_score(row):
    """
    Calculate location suitability score (0-100)
    Based on multiple factors
    """
    score = 0
    
    # 1. Competition factor (30 points)
    # Lower competition is better, but 0 competition might mean no demand
    comp_count = row['competitor_count']
    if comp_count == 0:
        score += 15  # No competition, but risky
    elif 1 <= comp_count <= 3:
        score += 30  # Sweet spot
    elif 4 <= comp_count <= 6:
        score += 20  # Moderate competition
    else:
        score += 10  # High competition
    
    # 2. Population factor (25 points)
    # Higher population = more potential customers
    pop = row['population']
    if pop > 100000:
        score += 25
    elif pop > 50000:
        score += 20
    elif pop > 25000:
        score += 15
    else:
        score += 5
    
    # 3. Road accessibility (20 points)
    # Close to major roads is better
    major_dist = row['major_road_dist_m']
    if major_dist < 300:
        score += 20
    elif major_dist < 600:
        score += 15
    elif major_dist < 1000:
        score += 10
    else:
        score += 5
    
    # 4. Rent factor (15 points)
    # Moderate rent is best (not too cheap, not too expensive)
    rent = row['rent_per_sqft']
    if 25 <= rent <= 60:
        score += 15  # Sweet spot
    elif 15 <= rent < 25:
        score += 10  # Cheap might mean bad location
    elif 60 < rent <= 100:
        score += 10  # Expensive but might be premium area
    else:
        score += 5
    
    # 5. Competitor distance (10 points)
    # Not too close to nearest competitor
    nearest = row['nearest_competitor_m']
    if nearest > 500:
        score += 10
    elif nearest > 300:
        score += 7
    elif nearest > 100:
        score += 5
    else:
        score += 2
    
    return min(100, score)  # Cap at 100


def generate_monthly_revenue(score, population, rent):
    """
    Generate synthetic monthly revenue based on score and features
    """
    # Base revenue from score (Rs 2-10 lakhs)
    base_revenue = 200000 + (score * 8000)
    
    # Population multiplier (0.7 - 1.5x)
    pop_multiplier = 0.7 + min(0.8, population / 200000)
    
    # Rent adjustment (inverse - higher rent = lower profit margin)
    rent_multiplier = max(0.8, 1.3 - (rent / 100))
    
    # Calculate revenue
    revenue = base_revenue * pop_multiplier * rent_multiplier
    
    # Add realistic noise (±15%)
    noise = np.random.normal(1.0, 0.15)
    revenue = revenue * noise
    
    # Ensure minimum revenue
    return max(100000, int(revenue))


def generate_success_label(score):
    """
    Generate binary success label
    Success = store survives and is profitable
    """
    # Base probability from score
    base_prob = score / 100
    
    # Add some randomness (±10%)
    noise = np.random.uniform(-0.1, 0.1)
    prob = max(0, min(1, base_prob + noise))
    
    # Generate binary label
    return 1 if np.random.random() < prob else 0


def generate_footfall_estimate(score, population):
    """
    Generate daily footfall estimate
    """
    # Base footfall from score (50-500 customers/day)
    base_footfall = 50 + (score * 4.5)
    
    # Population boost
    pop_boost = min(200, population / 1000)
    
    footfall = base_footfall + pop_boost
    
    # Add noise
    noise = np.random.normal(1.0, 0.2)
    footfall = footfall * noise
    
    return max(20, int(footfall))


def main():
    print("=" * 70)
    print("Generating Synthetic Training Labels")
    print("=" * 70)
    
    # Connect to database
    print("\n1. Connecting to database...")
    conn = connect_to_db()
    if not conn:
        print("Failed to connect to database. Exiting.")
        return
    
    # Fetch features
    print("\n2. Fetching grid cell features...")
    df = fetch_grid_features(conn)
    conn.close()
    
    if df.empty:
        print("No data found in database. Run feature_pipeline.py first.")
        return
    
    print(f"   Loaded {len(df)} grid cells")
    
    # Generate labels
    print("\n3. Generating synthetic labels...")
    
    # Calculate location scores
    df['location_score'] = df.apply(calculate_location_score, axis=1)
    
    # Generate revenue labels
    df['monthly_revenue'] = df.apply(
        lambda row: generate_monthly_revenue(
            row['location_score'], 
            row['population'], 
            row['rent_per_sqft']
        ), 
        axis=1
    )
    
    # Generate success labels
    df['success_label'] = df['location_score'].apply(generate_success_label)
    
    # Generate footfall estimates
    df['daily_footfall'] = df.apply(
        lambda row: generate_footfall_estimate(
            row['location_score'], 
            row['population']
        ), 
        axis=1
    )
    
    # Calculate profitability category
    df['profitability'] = pd.cut(
        df['monthly_revenue'],
        bins=[0, 200000, 400000, 600000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Save to CSV
    output_path = os.path.join('data', 'raw', 'synthetic_training_labels.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select columns to save
    output_df = df[[
        'cell_id', 'center_lat', 'center_lon',
        'location_score', 'monthly_revenue', 'success_label',
        'daily_footfall', 'profitability'
    ]]
    
    output_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total cells labeled: {len(df)}")
    print(f"\nLocation Score Statistics:")
    print(f"  Mean: {df['location_score'].mean():.2f}")
    print(f"  Median: {df['location_score'].median():.2f}")
    print(f"  Min: {df['location_score'].min():.2f}")
    print(f"  Max: {df['location_score'].max():.2f}")
    
    print(f"\nRevenue Statistics:")
    print(f"  Mean: ₹{df['monthly_revenue'].mean():,.0f}")
    print(f"  Median: ₹{df['monthly_revenue'].median():,.0f}")
    print(f"  Min: ₹{df['monthly_revenue'].min():,.0f}")
    print(f"  Max: ₹{df['monthly_revenue'].max():,.0f}")
    
    print(f"\nSuccess Rate: {df['success_label'].mean()*100:.1f}%")
    
    print(f"\nProfitability Distribution:")
    print(df['profitability'].value_counts().to_string())
    
    print(f"\nData saved to: {output_path}")
    print("=" * 70)
    
    print("\n⚠️  Note: These are SYNTHETIC labels for demonstration.")
    print("For production use, collect real store performance data.")


if __name__ == "__main__":
    main()
