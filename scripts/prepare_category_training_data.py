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


def add_fashion_specific_features(df):
    """Add features specifically useful for fashion/clothing stores"""
    print("\nAdding fashion-specific features...")
    
    # Income sensitivity (fashion is income-dependent)
    df['income_sensitivity'] = df['avg_monthly_income'] / (df['commercial_rent_per_sqft'] + 1)
    
    # Visibility importance (window shopping)
    df['visibility_score'] = df['road_density_km'] * df['footfall_accessibility_score']
    
    # Shopping district score (clustering effect)
    df['shopping_district_score'] = df['competitor_count'] * df['footfall_accessibility_score']
    
    # Parking accessibility (customers carry bags)
    df['parking_accessibility'] = df['transit_accessibility_score']
    
    # Market opportunity
    df['fashion_market_opportunity'] = df['purchasing_power_index'] / (df['competition_density'] + 1)
    
    return df


def add_electronics_specific_features(df):
    """Add features specifically useful for electronics stores"""
    print("\nAdding electronics-specific features...")
    
    # Tech hub proximity (IT parks, colleges)
    df['tech_hub_proximity'] = df['footfall_generator_count']
    
    # High-value customer base
    df['high_value_customer_score'] = df['avg_monthly_income'] * df['purchasing_power_index']
    
    # Showroom accessibility (customers research before buying)
    df['showroom_accessibility'] = df['connectivity_score'] * df['transit_accessibility_score']
    
    # Competition intensity (electronics is competitive)
    df['electronics_competition'] = df['competitor_count'] / (df['total_population'] / 1000 + 1)
    
    # Market potential
    df['electronics_market_potential'] = df['purchasing_power_index'] / (df['competition_density'] + 1)
    
    return df


def add_health_specific_features(df):
    """Add features specifically useful for health/pharmacy stores"""
    print("\nAdding health-specific features...")
    
    # Population density (more people = more customers)
    df['population_density'] = df['total_population'] / 1000
    
    # Accessibility importance (emergency access)
    df['emergency_accessibility'] = df['road_density_km'] + df['transit_accessibility_score']
    
    # Residential proximity (daily needs)
    df['residential_proximity'] = 1 / (df['distance_to_city_center'] + 1)
    
    # Competition (less competition is better for health)
    df['health_market_opportunity'] = df['total_population'] / (df['competitor_count'] + 1)
    
    # Elderly population proxy (income stability)
    df['stable_customer_base'] = df['avg_monthly_income']
    
    return df


def add_services_specific_features(df):
    """Add features specifically useful for service businesses (salons, gyms, etc.)"""
    print("\nAdding services-specific features...")
    
    # Residential + office mix (services need both)
    df['mixed_use_score'] = df['footfall_generator_count'] + (1 / (df['distance_to_city_center'] + 1))
    
    # Repeat customer potential (services rely on regulars)
    df['repeat_customer_potential'] = df['total_population'] / 1000
    
    # Convenience score (services need to be convenient)
    df['convenience_score'] = df['transit_accessibility_score'] + df['connectivity_score']
    
    # Income-dependent (premium services)
    df['premium_service_potential'] = df['avg_monthly_income'] * df['purchasing_power_index']
    
    # Low competition opportunity
    df['services_market_opportunity'] = df['total_population'] / (df['competitor_count'] + 1)
    
    return df



def split_by_category(df):
    """Split data by category and add category-specific features"""
    print("\nSplitting data by category...")
    
    categories = {}
    
    # Food
    food_data = df[df['main_category'] == 'food'].copy()
    if len(food_data) > 0:
        categories['food'] = add_food_specific_features(food_data)
    
    # Retail General
    retail_data = df[df['main_category'] == 'retail_general'].copy()
    if len(retail_data) > 0:
        categories['retail_general'] = add_retail_specific_features(retail_data)
    
    # Retail Fashion
    fashion_data = df[df['main_category'] == 'retail_fashion'].copy()
    if len(fashion_data) > 0:
        categories['retail_fashion'] = add_fashion_specific_features(fashion_data)
    
    # Retail Electronics
    electronics_data = df[df['main_category'] == 'retail_electronics'].copy()
    if len(electronics_data) > 0:
        categories['retail_electronics'] = add_electronics_specific_features(electronics_data)
    
    # Health
    health_data = df[df['main_category'] == 'health'].copy()
    if len(health_data) > 0:
        categories['health'] = add_health_specific_features(health_data)
    
    # Services
    services_data = df[df['main_category'] == 'services'].copy()
    if len(services_data) > 0:
        categories['services'] = add_services_specific_features(services_data)
    
    return categories


def save_category_data(categories):
    """Save category-specific datasets"""
    print("\nSaving category-specific datasets...")
    
    # Create directory if needed
    Path('data/processed/category_specific').mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    for category, data in categories.items():
        filename = f'data/processed/category_specific/training_data_{category}.csv'
        data.to_csv(filename, index=False)
        print(f"✓ Saved {category}: {len(data)} records, {len(data.columns)} features")
        
        # Print class distribution if success_label exists
        if 'success_label' in data.columns:
            success_rate = data['success_label'].mean()
            print(f"  Success rate: {success_rate:.2%}")



def main():
    print("=" * 70)
    print("PREPARING CATEGORY-SPECIFIC TRAINING DATA")
    print("=" * 70)
    
    # Load and merge
    training_data = load_and_merge_data()
    
    # Split by category and add features
    categories = split_by_category(training_data)
    
    # Save
    save_category_data(categories)
    
    print("\n" + "=" * 70)
    print("✓ PREPARATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

