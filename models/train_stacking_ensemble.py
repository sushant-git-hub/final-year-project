"""
Stacking ensemble combining XGBoost, LightGBM, and CatBoost
for category-specific predictions
"""
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pathlib import Path
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
    
    feature_names = X.columns.tolist()
    return X, y, feature_names

def load_optimized_params(category):
    """Load optimized hyperparameters if available"""
    params_path = Path(f'models/category_specific/optimized/{category}_best_params.json')
    
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"✓ Loaded optimized parameters for {category}")
        return params
    else:
        print(f"⚠️  No optimized parameters found for {category}, using defaults")
        return None

def create_base_models(category, optimized_params=None):
    """Create base models for stacking"""
    
    # XGBoost with optimized params
    if optimized_params:
        xgb_params = optimized_params.copy()
        xgb_params['random_state'] = 42
        xgb_params['tree_method'] = 'hist'
        xgb_params['eval_metric'] = 'logloss'
        xgb_model = xgb.XGBClassifier(**xgb_params)
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.03,
            random_state=42
        )
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1
    )
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=800,
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    
    return {
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'catboost': cat_model
    }

def train_stacking_ensemble(category):
    """Train stacking ensemble for a category"""
    print(f"\n{'='*70}")
    print(f"TRAINING STACKING ENSEMBLE: {category.upper()}")
    print(f"{'='*70}")
    
    # Load data
    X, y, feature_names = load_category_data(category)
    print(f"\nDataset: {len(X)} records, {len(feature_names)} features")
    print(f"Success rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_balanced)} records")
    
    # Load optimized params
    optimized_params = load_optimized_params(category)
    
    # Create base models
    print("\nCreating base models...")
    base_models = create_base_models(category, optimized_params)
    
    # Create stacking classifier
    estimators = [(name, model) for name, model in base_models.items()]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    # Train stacking model
    print("\nTraining stacking ensemble...")
    stacking_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\n--- Evaluation ---")
    
    # Training performance
    y_train_pred = stacking_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc:.1%})")
    
    # Test performance
    y_test_pred = stacking_model.predict(X_test)
    y_test_proba = stacking_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nTest Performance:")
    print(f"  Accuracy : {test_acc:.4f} ({test_acc:.1%})")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall   : {test_recall:.4f}")
    print(f"  F1-Score : {test_f1:.4f}")
    print(f"  AUC-ROC  : {test_auc:.4f}")
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Note: CV on stacking is expensive, so we'll use a simpler approach
    # Train individual models and average their CV scores
    cv_scores = []
    for name, model in base_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_scores.append(scores.mean())
        print(f"  {name:10s} CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    ensemble_cv = np.mean(cv_scores)
    print(f"  Ensemble (avg): {ensemble_cv:.4f}")
    
    # Save model
    save_dir = Path('models/category_specific/stacking')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f'{category}_stacking_model.pkl'
    joblib.dump(stacking_model, model_path)
    print(f"\n✓ Saved model to: {model_path}")
    
    # Save metrics
    metrics = {
        'category': category,
        'model_type': 'stacking_ensemble',
        'base_models': list(base_models.keys()),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'cv_mean': float(ensemble_cv),
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    metrics_path = save_dir / f'{category}_stacking_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    return stacking_model, metrics

def main():
    print("="*70)
    print("STACKING ENSEMBLE TRAINING")
    print("="*70)
    
    results = {}
    
    # Train retail stacking ensemble
    retail_model, retail_metrics = train_stacking_ensemble('retail')
    results['retail'] = retail_metrics
    
    # Train food stacking ensemble
    food_model, food_metrics = train_stacking_ensemble('food')
    results['food'] = food_metrics
    
    # Summary
    print("\n" + "="*70)
    print("STACKING ENSEMBLE SUMMARY")
    print("="*70)
    
    print(f"\nRetail Stacking Ensemble:")
    print(f"  Test Accuracy: {retail_metrics['test_accuracy']:.2%}")
    print(f"  CV Accuracy  : {retail_metrics['cv_mean']:.2%}")
    
    print(f"\nFood Stacking Ensemble:")
    print(f"  Test Accuracy: {food_metrics['test_accuracy']:.2%}")
    print(f"  CV Accuracy  : {food_metrics['cv_mean']:.2%}")
    
    # Compare with previous single models
    print(f"\n{'='*70}")
    print("COMPARISON WITH SINGLE MODELS")
    print(f"{'='*70}")
    
    print(f"\nRetail:")
    print(f"  Single XGBoost: 71.03% (test), 79.05% (CV)")
    print(f"  Stacking      : {retail_metrics['test_accuracy']:.2%} (test), {retail_metrics['cv_mean']:.2%} (CV)")
    improvement_retail = retail_metrics['test_accuracy'] - 0.7103
    print(f"  Improvement   : {improvement_retail:+.2%}")
    
    print(f"\nFood:")
    print(f"  Single XGBoost: 68.03% (test), 73.20% (CV)")
    print(f"  Stacking      : {food_metrics['test_accuracy']:.2%} (test), {food_metrics['cv_mean']:.2%} (CV)")
    improvement_food = food_metrics['test_accuracy'] - 0.6803
    print(f"  Improvement   : {improvement_food:+.2%}")
    
    # Overall weighted accuracy
    overall_stacking = (
        retail_metrics['test_accuracy'] * 0.93 +
        food_metrics['test_accuracy'] * 0.05 +
        0.6844 * 0.02
    )
    
    baseline = 0.6844
    single_category = 0.7083
    
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Baseline (Ensemble)        : {baseline:.2%}")
    print(f"Single Category Models     : {single_category:.2%} ({(single_category-baseline)*100:+.2f} pp)")
    print(f"Stacking Ensemble          : {overall_stacking:.2%} ({(overall_stacking-baseline)*100:+.2f} pp)")
    print(f"Improvement over Single    : {(overall_stacking-single_category)*100:+.2f} pp")
    
    if overall_stacking > single_category:
        print(f"\n✅ SUCCESS! Stacking ensemble improves over single models!")
    else:
        print(f"\n⚠️  Stacking did not improve over single models")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
