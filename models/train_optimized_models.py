"""
Retrain category-specific models with optimized hyperparameters
"""
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
    """Load optimized hyperparameters"""
    params_path = Path(f'models/category_specific/optimized/{category}_best_params.json')
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Add required parameters
    params['random_state'] = 42
    params['tree_method'] = 'hist'
    params['eval_metric'] = 'logloss'
    
    return params

def train_optimized_model(category):
    """Train model with optimized hyperparameters"""
    print(f"\n{'='*70}")
    print(f"TRAINING OPTIMIZED {category.upper()} MODEL")
    print(f"{'='*70}")
    
    # Load data
    X, y, feature_names = load_category_data(category)
    print(f"\nDataset: {len(X)} records, {len(feature_names)} features")
    print(f"Success rate: {y.mean():.2%}")
    
    # Load optimized parameters
    params = load_optimized_params(category)
    print(f"\nOptimized Parameters:")
    for param, value in params.items():
        if param not in ['random_state', 'tree_method', 'eval_metric']:
            print(f"  {param:20s}: {value}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_balanced)} records")
    
    # Train model
    print(f"\nTraining optimized {category} model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\n--- Evaluation ---")
    
    # Training performance
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc:.1%})")
    
    # Test performance
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
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
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model
    save_dir = Path('models/category_specific/optimized')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f'{category}_optimized_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Saved model to: {model_path}")
    
    # Save feature names
    features_path = save_dir / f'{category}_features.pkl'
    joblib.dump(feature_names, features_path)
    
    # Save metrics
    metrics = {
        'category': category,
        'model_type': 'optimized_xgboost',
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'hyperparameters': params
    }
    
    metrics_path = save_dir / f'{category}_optimized_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    return model, metrics

def main():
    print("="*70)
    print("RETRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("="*70)
    
    results = {}
    
    # Train optimized retail model
    retail_model, retail_metrics = train_optimized_model('retail')
    results['retail'] = retail_metrics
    
    # Train optimized food model
    food_model, food_metrics = train_optimized_model('food')
    results['food'] = food_metrics
    
    # Summary
    print("\n" + "="*70)
    print("OPTIMIZED MODELS SUMMARY")
    print("="*70)
    
    print(f"\nRetail Model:")
    print(f"  Test Accuracy: {retail_metrics['test_accuracy']:.2%}")
    print(f"  CV Accuracy  : {retail_metrics['cv_mean']:.2%} (+/- {retail_metrics['cv_std']:.2%})")
    
    print(f"\nFood Model:")
    print(f"  Test Accuracy: {food_metrics['test_accuracy']:.2%}")
    print(f"  CV Accuracy  : {food_metrics['cv_mean']:.2%} (+/- {food_metrics['cv_std']:.2%})")
    
    # Compare with previous models
    print(f"\n{'='*70}")
    print("COMPARISON WITH PREVIOUS MODELS")
    print(f"{'='*70}")
    
    print(f"\nRetail:")
    print(f"  Previous: 71.03% (test), 79.05% (CV)")
    print(f"  Optimized: {retail_metrics['test_accuracy']:.2%} (test), {retail_metrics['cv_mean']:.2%} (CV)")
    retail_improvement = retail_metrics['test_accuracy'] - 0.7103
    print(f"  Improvement: {retail_improvement:+.2%}")
    
    print(f"\nFood:")
    print(f"  Previous: 68.03% (test), 73.20% (CV)")
    print(f"  Optimized: {food_metrics['test_accuracy']:.2%} (test), {food_metrics['cv_mean']:.2%} (CV)")
    food_improvement = food_metrics['test_accuracy'] - 0.6803
    print(f"  Improvement: {food_improvement:+.2%}")
    
    # Overall weighted accuracy
    overall_optimized = (
        retail_metrics['test_accuracy'] * 0.93 +
        food_metrics['test_accuracy'] * 0.05 +
        0.6844 * 0.02
    )
    
    baseline = 0.6844
    previous_category = 0.7083
    
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Baseline (Ensemble)        : {baseline:.2%}")
    print(f"Previous Category Models   : {previous_category:.2%} ({(previous_category-baseline)*100:+.2f} pp)")
    print(f"Optimized Category Models  : {overall_optimized:.2%} ({(overall_optimized-baseline)*100:+.2f} pp)")
    print(f"Improvement over Previous  : {(overall_optimized-previous_category)*100:+.2f} pp")
    
    if overall_optimized > previous_category:
        print(f"\n✅ SUCCESS! Optimized models improve over previous models!")
    else:
        print(f"\n⚠️  Optimized models did not improve over previous models")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
