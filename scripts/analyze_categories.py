"""Analyze store categories for category-specific training"""
import pandas as pd
from collections import Counter

# Load data
df = pd.read_csv('data/raw/pune_all_retail_stores.csv')

# Parse types (they are comma-separated strings, not Python lists)
df['types_list'] = df['types'].apply(lambda x: [t.strip() for t in str(x).split(',')] if pd.notna(x) else [])

# Get all types
all_types = [t for types in df['types_list'] for t in types]

# Count
type_counts = Counter(all_types)

print("=" * 70)
print("STORE CATEGORY ANALYSIS")
print("=" * 70)
print(f"\nTotal stores: {len(df)}")
print(f"Total unique types: {len(type_counts)}")

print("\nTop 30 store types:")
for typ, count in type_counts.most_common(30):
    print(f"  {typ:40s}: {count:5d} stores")

# Identify main categories based on actual types in the data
main_categories = {
    'food': ['restaurant', 'cafe', 'bakery', 'meal_takeaway', 'meal_delivery'],
    'retail_general': ['store', 'supermarket', 'grocery_or_supermarket', 'shopping_mall'],
    'retail_fashion': ['clothing_store', 'shoe_store', 'jewelry_store'],
    'retail_electronics': ['electronics_store'],
    'health': ['pharmacy', 'drugstore', 'hospital', 'doctor', 'dentist'],
    'services': ['beauty_salon', 'hair_care', 'spa', 'gym', 'laundry'],
    'finance': ['atm', 'bank', 'finance'],
    'automotive': ['car_dealer', 'car_repair'],
}

# Categorize stores
def get_main_category(types_list):
    for category, keywords in main_categories.items():
        for typ in types_list:
            if any(keyword in typ.lower() for keyword in keywords):
                return category
    return 'other'

df['main_category'] = df['types_list'].apply(get_main_category)

print("\n" + "=" * 70)
print("MAIN CATEGORY DISTRIBUTION")
print("=" * 70)
print(df['main_category'].value_counts())

# Save categorized data
df[['place_id', 'name', 'types', 'main_category']].to_csv('data/raw/stores_with_categories.csv', index=False)
print("\n✓ Saved categorized stores to: data/raw/stores_with_categories.csv")
