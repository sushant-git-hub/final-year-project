"""
Category-specific hyperparameter optimization using Optuna
"""
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_category_data(category):
    """Load category-specific training data"""
    filepath = f'data/processed/category_specific/training_data_{category}.csv'
    df = pd.read_csv(filepath)
    
    # Columns to drop
    cols_to_drop = [
        'place_id', 'name', 'success_label', 'main_category',
        'cell_id', 'ward_id', 'ward_name', 'locality',
        'footfall_category', 'profitability', 'rental_zone',
        'ward_zone', 'income_tier', 'tier', 'confidence',
        'rating', 'location_score', 'daily_footfall', 'monthly_revenue'
    ]
    
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    y = df['success_label']
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def objective_retail(trial):
    """Optuna objective for retail model"""
    # Load data
    X, y = load_category_data('retail')
    
    # Hyperparameter search space (optimized for large dataset)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
        'max_depth': trial.suggest_int('max_depth', 7, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'gamma': trial.suggest_float('gamma', 0.1, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Cross-validation
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()

def objective_food(trial):
    """Optuna objective for food model"""
    # Load data
    X, y = load_category_data('food')
    
    # Hyperparameter search space (optimized for small dataset - more regularization)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 0.85),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
        'gamma': trial.suggest_float('gamma', 0.2, 0.6),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.5, 4.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.3),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Cross-validation
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()

def optimize_category(category, n_trials=100):
    """Optimize hyperparameters for a category"""
    print(f"\n{'='*70}")
    print(f"OPTIMIZING {category.upper()} MODEL")
    print(f"{'='*70}")
    print(f"Running {n_trials} trials with Optuna...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{category}_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    objective = objective_retail if category == 'retail' else objective_food
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print(f"\n--- Optimization Results ---")
    print(f"Best CV Accuracy: {study.best_value:.4f} ({study.best_value:.2%})")
    print(f"\nBest Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param:20s}: {value}")
    
    # Save best params
    import json
    from pathlib import Path
    
    save_dir = Path('models/category_specific/optimized')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    params_path = save_dir / f'{category}_best_params.json'
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n✓ Saved best parameters to: {params_path}")
    
    return study.best_params, study.best_value

def main():
    print("="*70)
    print("CATEGORY-SPECIFIC HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Optimize retail model
    retail_params, retail_score = optimize_category('retail', n_trials=100)
    
    # Optimize food model
    food_params, food_score = optimize_category('food', n_trials=100)
    
    # Summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nRetail Model:")
    print(f"  Best CV Accuracy: {retail_score:.2%}")
    print(f"  Previous CV Accuracy: 79.05%")
    print(f"  Improvement: {(retail_score - 0.7905)*100:+.2f} percentage points")
    
    print(f"\nFood Model:")
    print(f"  Best CV Accuracy: {food_score:.2%}")
    print(f"  Previous CV Accuracy: 73.20%")
    print(f"  Improvement: {(food_score - 0.7320)*100:+.2f} percentage points")
    
    # Estimate overall improvement
    overall_optimized = retail_score * 0.93 + food_score * 0.05 + 0.6844 * 0.02
    overall_previous = 0.7855  # Previous CV accuracy
    
    print(f"\nOverall (Weighted):")
    print(f"  Optimized CV Accuracy: {overall_optimized:.2%}")
    print(f"  Previous CV Accuracy: {overall_previous:.2%}")
    print(f"  Improvement: {(overall_optimized - overall_previous)*100:+.2f} percentage points")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
