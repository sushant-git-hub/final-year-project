"""
Prepare category-specific training data for specialized models
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_merge_data():
    """Load training data and merge with category information"""
    print("Loading data...")
    
    # Load training data
    training_data = pd.read_csv('data/processed/training_data.csv')
    print(f"Training data: {len(training_data)} records")
    
    # Load category mapping
    categories = pd.read_csv('data/raw/stores_with_categories.csv')
    print(f"Category data: {len(categories)} records")
    
    # Merge categories
    training_data = training_data.merge(
        categories[['place_id', 'main_category']], 
        on='place_id', 
        how='left'
    )
    
    # Fill missing categories with 'other'
    training_data['main_category'] = training_data['main_category'].fillna('other')
    
    print("\nCategory distribution in training data:")
    print(training_data['main_category'].value_counts())
    
    return training_data

def add_retail_specific_features(df):
    """Add features specifically useful for retail stores"""
    print("\nAdding retail-specific features...")
    
    # Competition intensity
    df['retail_competition_score'] = df['competitor_count'] / (df['footfall_accessibility_score'] + 1)
    
    # Rent affordability
    df['rent_affordability'] = df['commercial_rent_per_sqft'] / (df['avg_monthly_income'] + 1)
    
    # Parking accessibility (using transit as proxy since parking data may not be separate)
    df['parking_accessibility'] = df['transit_accessibility_score']
    
    # Retail density (competition per area)
    df['retail_density'] = df['competition_density']
    
    # Visibility score (combination of road density and footfall)
    df['visibility_score'] = df['road_density_km'] * df['footfall_accessibility_score']
    
    # Market opportunity (low competition + high footfall)
    df['market_opportunity'] = df['footfall_accessibility_score'] / (df['competition_density'] + 1)
    
    return df

def add_food_specific_features(df):
    """Add features specifically useful for food/restaurant stores"""
    print("\nAdding food-specific features...")
    
    # Residential proximity (inverse of distance to city center as proxy)
    df['residential_proximity'] = 1 / (df['distance_to_city_center'] + 1)
    
    # Office accessibility (using footfall generators as proxy for offices/colleges)
    df['office_accessibility'] = df['footfall_generator_count']
    
    # Transit accessibility (already available)
    df['transit_access'] = df['transit_accessibility_score']
    
    # Evening potential (residential + footfall)
    df['evening_potential'] = df['residential_proximity'] * df['footfall_accessibility_score']
    
    # Lunch crowd potential (transit + footfall generators)
    df['lunch_potential'] = df['transit_accessibility_score'] * df['footfall_generator_count']
    
    # Connectivity importance for food (people travel for good food)
    df['food_connectivity'] = df['connectivity_score'] * df['transit_accessibility_score']
    
    return df


def split_by_category(df):
    """Split data by category"""
    print("\nSplitting data by category...")
    
    # Retail data
    retail_data = df[df['main_category'] == 'retail_general'].copy()
    retail_data = add_retail_specific_features(retail_data)
    
    # Food data
    food_data = df[df['main_category'] == 'food'].copy()
    food_data = add_food_specific_features(food_data)
    
    # Other categories (use general features)
    other_data = df[~df['main_category'].isin(['retail_general', 'food'])].copy()
    
    return retail_data, food_data, other_data

def save_category_data(retail_data, food_data, other_data):
    """Save category-specific datasets"""
    print("\nSaving category-specific datasets...")
    
    # Create directory if needed
    Path('data/processed/category_specific').mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    retail_data.to_csv('data/processed/category_specific/training_data_retail.csv', index=False)
    food_data.to_csv('data/processed/category_specific/training_data_food.csv', index=False)
    other_data.to_csv('data/processed/category_specific/training_data_other.csv', index=False)
    
    print(f"\n✓ Saved retail data: {len(retail_data)} records")
    print(f"✓ Saved food data: {len(food_data)} records")
    print(f"✓ Saved other data: {len(other_data)} records")
    
    # Print feature counts
    print(f"\nRetail features: {len(retail_data.columns)}")
    print(f"Food features: {len(food_data.columns)}")
    print(f"Other features: {len(other_data.columns)}")
    
    # Print class distribution
    print("\n--- Class Distribution ---")
    print(f"Retail - Success rate: {retail_data['success_label'].mean():.2%}")
    print(f"Food - Success rate: {food_data['success_label'].mean():.2%}")
    print(f"Other - Success rate: {other_data['success_label'].mean():.2%}")


def main():
    print("=" * 70)
    print("PREPARING CATEGORY-SPECIFIC TRAINING DATA")
    print("=" * 70)
    
    # Load and merge
    training_data = load_and_merge_data()
    
    # Split by category and add features
    retail_data, food_data, other_data = split_by_category(training_data)
    
    # Save
    save_category_data(retail_data, food_data, other_data)
    
    print("\n" + "=" * 70)
    print("✓ PREPARATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
