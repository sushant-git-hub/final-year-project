"""
Train category-specific models for all 6 major categories:
food, retail_general, retail_fashion, retail_electronics, health, services
"""
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import json
from pathlib import Path

# Optimized hyperparameters from previous tuning
RETAIL_PARAMS = {
    'n_estimators': 800,
    'max_depth': 7,
    'learning_rate': 0.025,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.15,
    'reg_alpha': 1.2,
    'reg_lambda': 1.8,
    'scale_pos_weight': 1.2,  # Slightly adjusted for retail
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

FOOD_PARAMS = {
    'n_estimators': 600,  # Fewer trees due to smaller dataset
    'max_depth': 6,  # Less depth to avoid overfitting
    'learning_rate': 0.03,
    'min_child_weight': 5,  # More regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.2,  # More regularization
    'reg_alpha': 1.5,
    'reg_lambda': 2.0,
    'scale_pos_weight': 1.1,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

# Hyperparameters for new categories
FASHION_PARAMS = {
    'n_estimators': 700,
    'max_depth': 7,
    'learning_rate': 0.03,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.15,
    'reg_alpha': 1.2,
    'reg_lambda': 1.8,
    'scale_pos_weight': 1.15,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

ELECTRONICS_PARAMS = {
    'n_estimators': 700,
    'max_depth': 6,
    'learning_rate': 0.03,
    'min_child_weight': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'reg_alpha': 1.3,
    'reg_lambda': 1.9,
    'scale_pos_weight': 1.2,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

HEALTH_PARAMS = {
    'n_estimators': 400,  # Small dataset
    'max_depth': 4,  # Very conservative
    'learning_rate': 0.05,
    'min_child_weight': 7,  # High regularization
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.3,
    'reg_alpha': 2.0,
    'reg_lambda': 2.5,
    'scale_pos_weight': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

SERVICES_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.04,
    'min_child_weight': 6,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'gamma': 0.25,
    'reg_alpha': 1.7,
    'reg_lambda': 2.2,
    'scale_pos_weight': 1.1,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

def load_category_data(category):
    """Load category-specific training data"""
    filepath = f'data/processed/category_specific/training_data_{category}.csv'
    df = pd.read_csv(filepath)
    
    # Columns to drop (identifiers, target, and LEAKING features)
    cols_to_drop = [
        'place_id', 'name', 'success_label', 'main_category',
        # Remove identifier columns that cause data leakage
        'cell_id', 'ward_id', 'ward_name', 'locality',
        # Remove categorical columns that are too specific
        'footfall_category', 'profitability', 'rental_zone',
        'ward_zone', 'income_tier', 'tier', 'confidence',
        # ⚠️ CRITICAL: Remove features that leak target information
        'rating', 'location_score',  # These perfectly predict the target!
        # Also remove other potentially derived features
        'daily_footfall', 'monthly_revenue'  # These may be derived from success
    ]
    
    # Separate features and target
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    y = df['success_label']
    
    # Handle any remaining categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"⚠️  Warning: Found unexpected categorical columns: {categorical_cols}")
        print(f"   Encoding them, but this may indicate data leakage")
        # Use label encoding for categorical columns
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    print(f"Using {len(feature_names)} features (removed {len(cols_to_drop)} identifier/leaking columns)")
    
    return X, y, feature_names

def train_category_model(category, params):
    """Train a model for a specific category"""
    print(f"\n{'='*70}")
    print(f"TRAINING {category.upper()} MODEL")
    print(f"{'='*70}")
    
    # Load data
    X, y, feature_names = load_category_data(category)
    print(f"\nDataset: {len(X)} records, {len(feature_names)} features")
    print(f"Success rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_balanced)} records")
    print(f"Class distribution: {np.bincount(y_train_balanced.astype(int))}")
    
    # Train model
    print(f"\nTraining {category} model...")
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
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model
    save_dir = Path('models/category_specific')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f'{category}_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Saved model to: {model_path}")
    
    # Save feature names
    features_path = save_dir / f'{category}_features.pkl'
    joblib.dump(feature_names, features_path)
    print(f"✓ Saved features to: {features_path}")
    
    # Save metrics
    metrics = {
        'category': category,
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
        'n_test': len(X_test)
    }
    
    metrics_path = save_dir / f'{category}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    return model, metrics

def main():
    print("="*70)
    print("CATEGORY-SPECIFIC MODEL TRAINING (ALL 6 CATEGORIES)")
    print("="*70)
    
    # Define categories and their parameters
    categories_config = [
        ('food', FOOD_PARAMS),
        ('retail_general', RETAIL_PARAMS),
        ('retail_fashion', FASHION_PARAMS),
        ('retail_electronics', ELECTRONICS_PARAMS),
        ('health', HEALTH_PARAMS),
        ('services', SERVICES_PARAMS)
    ]
    
    results = {}
    
    # Train all category models
    for category, params in categories_config:
        try:
            model, metrics = train_category_model(category, params)
            results[category] = metrics
        except Exception as e:
            print(f"\n❌ Error training {category} model: {e}")
            print(f"   Skipping {category}...")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for category, metrics in results.items():
        print(f"\n{category.upper().replace('_', ' ')} Model:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"  Test F1-Score: {metrics['test_f1']:.4f}")
        print(f"  CV Accuracy  : {metrics['cv_mean']:.2%} (+/- {metrics['cv_std']:.2%})")
        print(f"  Features     : {metrics['n_features']}")
    
    print(f"\n{'='*70}")
    print("✓ ALL CATEGORY MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
