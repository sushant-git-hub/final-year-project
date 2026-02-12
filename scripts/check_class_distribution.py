import pandas as pd

# Check retail data
retail = pd.read_csv('data/processed/category_specific/training_data_retail.csv')
print('RETAIL DATA:')
print(f'  Total records: {len(retail)}')
print(f'  Success (1): {retail["success_label"].sum()}')
print(f'  Failure (0): {(1-retail["success_label"]).sum()}')
print(f'  Success rate: {retail["success_label"].mean():.2%}')

# Check food data
food = pd.read_csv('data/processed/category_specific/training_data_food.csv')
print('\nFOOD DATA:')
print(f'  Total records: {len(food)}')
print(f'  Success (1): {food["success_label"].sum()}')
print(f'  Failure (0): {(1-food["success_label"]).sum()}')
print(f'  Success rate: {food["success_label"].mean():.2%}')

# Check original training data
original = pd.read_csv('data/processed/training_data.csv')
print('\nORIGINAL TRAINING DATA:')
print(f'  Total records: {len(original)}')
print(f'  Success (1): {original["success_label"].sum()}')
print(f'  Failure (0): {(1-original["success_label"]).sum()}')
print(f'  Success rate: {original["success_label"].mean():.2%}')
