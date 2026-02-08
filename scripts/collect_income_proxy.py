"""
Income Level Proxy Data Collection
Uses property prices and area characteristics to estimate income levels
"""

import pandas as pd
import numpy as np
import os


def load_ward_data():
    """Load existing ward data"""
    ward_path = os.path.join('data', 'raw', 'pune_wards_for_postgis.csv')
    if not os.path.exists(ward_path):
        print(f"Error: Ward data not found at {ward_path}")
        return None
    
    return pd.read_csv(ward_path)


def estimate_income_by_area(ward_name, zone):
    """
    Estimate income tier based on area characteristics
    Based on known high-income, mid-income, and low-income areas in Pune
    """
    ward_name_lower = str(ward_name).lower()
    
    # High-income areas in Pune
    high_income_areas = [
        'koregaon park', 'kalyani nagar', 'baner', 'aundh', 'boat club',
        'hinjewadi', 'magarpatta', 'viman nagar', 'kharadi', 'wakad',
        'bavdhan', 'sus', 'pashan', 'model colony'
    ]
    
    # Mid-income areas
    mid_income_areas = [
        'kothrud', 'shivajinagar', 'deccan', 'sadashiv peth', 'fc road',
        'camp', 'pimpri', 'chinchwad', 'nigdi', 'akurdi', 'warje',
        'karve nagar', 'erandwane', 'paud road', 'sahakarnagar'
    ]
    
    # Check for high-income areas
    for area in high_income_areas:
        if area in ward_name_lower:
            return {
                'income_tier': 'High',
                'avg_monthly_income': 75000,
                'min_income': 50000,
                'max_income': 150000,
                'property_price_sqft': 8000
            }
    
    # Check for mid-income areas
    for area in mid_income_areas:
        if area in ward_name_lower:
            return {
                'income_tier': 'Medium',
                'avg_monthly_income': 45000,
                'min_income': 30000,
                'max_income': 70000,
                'property_price_sqft': 5000
            }
    
    # Default to low-income
    return {
        'income_tier': 'Low',
        'avg_monthly_income': 25000,
        'min_income': 15000,
        'max_income': 40000,
        'property_price_sqft': 3000
    }


def estimate_purchasing_power(income_tier, population):
    """
    Estimate purchasing power index based on income and population
    """
    tier_multipliers = {
        'High': 1.5,
        'Medium': 1.0,
        'Low': 0.6
    }
    
    multiplier = tier_multipliers.get(income_tier, 1.0)
    
    # Purchasing power = (population / 1000) * income_multiplier
    purchasing_power = (population / 1000) * multiplier
    
    return round(purchasing_power, 2)


def main():
    print("=" * 70)
    print("Generating Income Proxy Data for Pune Wards")
    print("=" * 70)
    
    # Load ward data
    print("\n1. Loading ward data...")
    wards_df = load_ward_data()
    
    if wards_df is None:
        return
    
    print(f"   Loaded {len(wards_df)} wards")
    
    # Estimate income for each ward
    print("\n2. Estimating income levels based on area characteristics...")
    
    income_data = []
    for _, row in wards_df.iterrows():
        ward_name = row.get('ward_name', '')
        zone = row.get('zone', '')
        population = row.get('total_population', 0)
        
        income_info = estimate_income_by_area(ward_name, zone)
        purchasing_power = estimate_purchasing_power(
            income_info['income_tier'], 
            population
        )
        
        income_data.append({
            'ward_id': row.get('ward_id', ''),
            'ward_name': ward_name,
            'zone': zone,
            'latitude': row.get('latitude', 0),
            'longitude': row.get('longitude', 0),
            'total_population': population,
            'income_tier': income_info['income_tier'],
            'avg_monthly_income': income_info['avg_monthly_income'],
            'min_income': income_info['min_income'],
            'max_income': income_info['max_income'],
            'property_price_sqft': income_info['property_price_sqft'],
            'purchasing_power_index': purchasing_power
        })
    
    # Create DataFrame
    income_df = pd.DataFrame(income_data)
    
    # Save to CSV
    output_path = os.path.join('data', 'raw', 'pune_income_proxy.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    income_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total wards processed: {len(income_df)}")
    print(f"\nIncome tier distribution:")
    print(income_df['income_tier'].value_counts().to_string())
    print(f"\nAverage income by tier:")
    print(income_df.groupby('income_tier')['avg_monthly_income'].mean().to_string())
    print(f"\nData saved to: {output_path}")
    print("=" * 70)
    
    print("\n⚠️  Note: This is proxy data based on area characteristics.")
    print("For more accurate income data, consider:")
    print("  1. Census of India data (censusindia.gov.in)")
    print("  2. Market research reports")
    print("  3. Property price scraping from 99acres/MagicBricks")


if __name__ == "__main__":
    main()
