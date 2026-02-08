"""
Generate REAL training labels from existing store data
Uses actual Google Places ratings and review counts instead of synthetic data
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_store_data():
    """Load existing retail store data with ratings"""
    stores_file = project_root / "data" / "raw" / "pune_all_retail_stores.csv"
    
    if not stores_file.exists():
        print(f"❌ Error: Store data not found at {stores_file}")
        return None
    
    print(f"📂 Loading store data from {stores_file}")
    df = pd.read_csv(stores_file)
    
    # Filter for operational stores with ratings
    df_clean = df[
        (df['business_status'] == 'OPERATIONAL') &
        (df['rating'].notna()) &
        (df['user_ratings_total'] > 0)
    ].copy()
    
    print(f"   ✓ Loaded {len(df)} total stores")
    print(f"   ✓ Filtered to {len(df_clean)} operational stores with ratings")
    
    return df_clean

def create_success_label(row):
    """
    Create binary success label based on rating
    High rating (>= 4.0) = successful location
    """
    if pd.isna(row['rating']):
        return 0
    return 1 if row['rating'] >= 4.0 else 0

def estimate_footfall_category(user_ratings_total):
    """
    Estimate footfall category based on number of reviews
    More reviews = more customers
    """
    if user_ratings_total >= 100:
        return 'High'
    elif user_ratings_total >= 20:
        return 'Medium'
    else:
        return 'Low'

def estimate_daily_footfall(user_ratings_total, rating):
    """
    Estimate daily footfall based on reviews and rating
    Assumption: 1 review per 50-100 customers (conservative)
    """
    # Base footfall from review count
    base_footfall = user_ratings_total * 75  # Assume 75 customers per review
    
    # Adjust by rating (higher rating = more repeat customers)
    rating_multiplier = rating / 5.0 if not pd.isna(rating) else 0.8
    
    # Convert to daily estimate (assume reviews accumulated over 2 years)
    daily_footfall = (base_footfall * rating_multiplier) / (365 * 2)
    
    return int(max(daily_footfall, 10))  # Minimum 10 customers/day

def estimate_monthly_revenue(daily_footfall, rating, store_type):
    """
    Estimate monthly revenue based on footfall, rating, and store type
    """
    # Average transaction value by store type (in INR)
    avg_transaction = {
        'supermarket': 500,
        'grocery': 300,
        'clothing': 1200,
        'pharmacy': 250,
        'bakery': 150,
        'restaurant': 400,
        'electronics': 3000,
        'jewelry': 8000,
        'default': 500
    }
    
    # Determine store type from types column
    transaction_value = avg_transaction.get('default', 500)
    for key in avg_transaction.keys():
        if key in store_type.lower():
            transaction_value = avg_transaction[key]
            break
    
    # Calculate monthly revenue
    # Assume 30% of footfall makes a purchase
    conversion_rate = 0.3
    monthly_customers = daily_footfall * 30 * conversion_rate
    monthly_revenue = monthly_customers * transaction_value
    
    # Adjust by rating
    rating_factor = rating / 5.0 if not pd.isna(rating) else 0.8
    monthly_revenue *= rating_factor
    
    return int(monthly_revenue)

def calculate_profitability_category(monthly_revenue):
    """Categorize profitability based on revenue"""
    if monthly_revenue >= 500000:
        return 'High'
    elif monthly_revenue >= 200000:
        return 'Medium'
    else:
        return 'Low'

def main():
    print("=" * 70)
    print("GENERATING REAL TRAINING LABELS FROM STORE DATA")
    print("=" * 70)
    
    # Load store data
    print("\n1. Loading existing store data...")
    stores_df = load_store_data()
    
    if stores_df is None:
        return
    
    print(f"\n2. Creating training labels from real data...")
    
    # Create labels
    labels_data = []
    
    for _, row in stores_df.iterrows():
        # Extract store info
        place_id = row['place_id']
        name = row['name']
        lat = row['latitude']
        lon = row['longitude']
        rating = row['rating']
        user_ratings_total = row['user_ratings_total']
        store_type = row['types'] if pd.notna(row['types']) else 'default'
        
        # Generate labels
        success_label = create_success_label(row)
        daily_footfall = estimate_daily_footfall(user_ratings_total, rating)
        footfall_category = estimate_footfall_category(user_ratings_total)
        monthly_revenue = estimate_monthly_revenue(daily_footfall, rating, store_type)
        profitability = calculate_profitability_category(monthly_revenue)
        
        # Location score (0-100) based on rating
        location_score = (rating / 5.0) * 100 if not pd.isna(rating) else 50
        
        labels_data.append({
            'place_id': place_id,
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'rating': rating,
            'user_ratings_total': user_ratings_total,
            'location_score': round(location_score, 2),
            'success_label': success_label,
            'daily_footfall': daily_footfall,
            'footfall_category': footfall_category,
            'monthly_revenue': monthly_revenue,
            'profitability': profitability
        })
    
    # Create DataFrame
    labels_df = pd.DataFrame(labels_data)
    
    # Save to CSV
    output_file = project_root / "data" / "raw" / "real_training_labels.csv"
    labels_df.to_csv(output_file, index=False)
    
    print(f"   ✓ Created {len(labels_df)} training labels")
    
    # Print summary statistics
    print("\n3. Summary Statistics:")
    print(f"   • Success Rate: {labels_df['success_label'].mean():.1%}")
    print(f"   • Average Rating: {labels_df['rating'].mean():.2f}")
    print(f"   • Average Reviews: {labels_df['user_ratings_total'].mean():.0f}")
    print(f"   • Average Daily Footfall: {labels_df['daily_footfall'].mean():.0f}")
    print(f"   • Average Monthly Revenue: ₹{labels_df['monthly_revenue'].mean():,.0f}")
    
    print("\n   Footfall Distribution:")
    print(labels_df['footfall_category'].value_counts())
    
    print("\n   Profitability Distribution:")
    print(labels_df['profitability'].value_counts())
    
    print(f"\n✅ COMPLETE! Saved to: {output_file}")
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Use this data for ML model training")
    print("2. Join with grid features from feature_pipeline.py")
    print("3. Train model to predict success_label or monthly_revenue")
    print("=" * 70)

if __name__ == "__main__":
    main()
