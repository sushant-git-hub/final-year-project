"""
Hyperparameter Optimization using Optuna
----------------------------------------
Automatically finds best hyperparameters for XGBoost to maximize accuracy.

Usage:
    python models/optimize_hyperparameters.py
    
This will run for ~2-4 hours. Let it run overnight for best results.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.model_config import get_feature_columns

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_and_prepare_data():
    """Load and prepare data."""
    print("📂 Loading training data...")
    data_path = project_root / "data" / "processed" / "training_data.csv"
    df = pd.read_csv(data_path)
    
    # Prepare features
    feature_cols = get_feature_columns(df)
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df['success_label'].copy()
    
    # Handle categoricals
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X = X.fillna(0)
    
    # Feature selection
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
    
    selector_var = VarianceThreshold(threshold=0.01)
    X_var = selector_var.fit_transform(X)
    features_var = X.columns[selector_var.get_support()].tolist()
    X = pd.DataFrame(X_var, columns=features_var)
    
    selector_k = SelectKBest(f_classif, k=min(100, len(features_var)))
    X_selected = selector_k.fit_transform(X, y)
    features_selected = X.columns[selector_k.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=features_selected)
    
    print(f"   ✓ Loaded {len(df)} samples with {len(X.columns)} features")
    
    return X, y, X.columns.tolist()


def objective(trial, X, y):
    """Optuna objective function."""
    
    # Hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()


def main():
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 70)
    print("\n⚠️  This will take 2-4 hours. Let it run overnight for best results.")
    print("   You can stop it anytime with Ctrl+C and use the best params so far.\n")
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Apply SMOTE
    print("\n⚖️  Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"   ✓ Balanced: {len(X)} → {len(X_balanced)} samples")
    
    # Create Optuna study
    print("\n🔍 Starting hyperparameter optimization...")
    print("   Target: 100 trials (can stop early with Ctrl+C)")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    try:
        study.optimize(
            lambda trial: objective(trial, X_balanced, y_balanced),
            n_trials=100,
            show_progress_bar=True,
            n_jobs=1  # Use 1 job since cross_val_score uses -1
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n✅ Best Accuracy: {study.best_value:.4f} ({study.best_value:.1%})")
    print(f"✅ Total Trials: {len(study.trials)}")
    
    print("\n📊 BEST HYPERPARAMETERS:")
    for param, value in study.best_params.items():
        print(f"   {param:20s}: {value}")
    
    # Save best params
    output_dir = project_root / "models" / "optimized"
    output_dir.mkdir(exist_ok=True)
    
    best_params_path = output_dir / "best_xgboost_params.json"
    import json
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n💾 Saved best parameters to: {best_params_path}")
    
    # Train final model with best params
    print("\n🚀 Training final model with optimized parameters...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    best_model = xgb.XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n   ✅ Test Accuracy: {test_acc:.4f} ({test_acc:.1%})")
    print(f"   ✅ Test AUC-ROC: {test_auc:.4f}")
    
    # Save model
    model_path = output_dir / "xgb_optimized_model.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(feature_names, output_dir / "feature_names.pkl")
    
    print(f"\n💾 Saved optimized model to: {model_path}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("=" * 70)
    print("   Original XGBoost    : 54.6%")
    print("   Improved XGBoost    : 66.5%")
    print("   Voting Ensemble     : 68.4%")
    print(f"   Optimized XGBoost   : {test_acc:.1%}")
    print(f"   Improvement         : {(test_acc - 0.684) * 100:+.1f}% vs Ensemble")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Use these parameters in your training scripts")
    print("2. Create ensemble with optimized models")
    print("3. Expected final accuracy: 70-75%")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n\n❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
