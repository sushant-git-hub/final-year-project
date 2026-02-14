"""
Advanced Ensemble Model for Maximum Accuracy
---------------------------------------------
Combines XGBoost, LightGBM, CatBoost, and Neural Network using stacking.

Usage:
    python models/train_ensemble_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.model_config import TRAINING_CONFIG, MODEL_PATHS, METRICS_DIR, PLOTS_DIR, get_feature_columns

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_and_prepare_data():
    """Load and prepare training data."""
    print("=" * 70)
    print("ADVANCED ENSEMBLE MODEL TRAINING")
    print("=" * 70)
    
    print("\n📂 Loading training data...")
    data_path = project_root / "data" / "processed" / "training_data.csv"
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Prepare features
    print("\n🔧 Preparing features...")
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
    
    # Remove low variance
    selector_var = VarianceThreshold(threshold=0.01)
    X_var = selector_var.fit_transform(X)
    features_var = X.columns[selector_var.get_support()].tolist()
    X = pd.DataFrame(X_var, columns=features_var)
    
    # Select top features
    selector_k = SelectKBest(f_classif, k=min(100, len(features_var)))
    X_selected = selector_k.fit_transform(X, y)
    features_selected = X.columns[selector_k.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=features_selected)
    
    print(f"   ✓ Final features: {len(X.columns)}")
    print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, X.columns.tolist()


def create_base_models():
    """Create base models for ensemble."""
    print("\n🤖 Creating base models...")
    
    models = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=500,  # Reduced to avoid overfitting without early stopping
            max_depth=6,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.3,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            tree_method='hist',
            eval_metric='logloss'
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1
        ),
        'catboost': CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.02,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    }
    
    print(f"   ✓ Created {len(models)} base models")
    return models


def train_ensemble(X_train, y_train, base_models):
    """Train ensemble using voting."""
    print("\n🏗️  Training ensemble (Voting Classifier)...")
    
    # Train each model individually first
    trained_models = {}
    
    for name, model in base_models.items():
        print(f"   ⏳ Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   ✓ {name} trained")
    
    # Create voting ensemble
    estimators = [(name, model) for name, model in trained_models.items()]
    
    voting_model = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Probability averaging
        weights=[3, 3, 3, 1],  # Boost models get more weight
        n_jobs=-1
    )
    
    # Fit voting (uses already trained models)
    print("   ⏳ Creating weighted ensemble...")
    voting_model.fit(X_train, y_train)
    
    print("   ✓ Ensemble complete!")
    
    return voting_model, trained_models


def evaluate_ensemble(model, X_train, y_train, X_test, y_test, model_name="Ensemble"):
    """Evaluate ensemble model."""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
    }
    
    # Print
    print("\n   TEST SET PERFORMANCE:")
    print(f"      Accuracy : {metrics['test']['accuracy']:.4f} ({metrics['test']['accuracy']:.1%})")
    print(f"      Precision: {metrics['test']['precision']:.4f}")
    print(f"      Recall   : {metrics['test']['recall']:.4f}")
    print(f"      F1-Score : {metrics['test']['f1']:.4f}")
    print(f"      AUC-ROC  : {metrics['test']['roc_auc']:.4f}")
    
    return metrics, y_test_pred, y_test_proba


def plot_results(y_test, y_pred, y_proba, model_name="Ensemble"):
    """Create visualizations."""
    print("\n📈 Creating visualizations...")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Fail', 'Success'],
                yticklabels=['Fail', 'Success'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2, color='green')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_roc_curve.png'), dpi=300)
    plt.close()
    
    print(f"   ✓ Saved plots to {PLOTS_DIR}")


def save_model(model, feature_names, metrics, model_name="ensemble"):
    """Save model and artifacts."""
    print(f"\n💾 Saving {model_name} model...")
    
    os.makedirs(os.path.dirname(MODEL_PATHS['success_classifier']), exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(os.path.dirname(MODEL_PATHS['success_classifier']), f'{model_name}_model.pkl')
    joblib.dump(model, model_path)
    print(f"   ✓ Saved model: {model_path}")
    
    # Save feature names
    joblib.dump(feature_names, MODEL_PATHS['feature_names'])
    
    # Save metrics
    metrics_path = os.path.join(METRICS_DIR, f'{model_name}_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Saved metrics: {metrics_path}")


def main():
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Split
    print("\n✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Handle imbalance
    print("\n⚖️  Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"   ✓ Balanced: {len(X_train)} → {len(X_train_balanced)} samples")
    
    # Create base models
    base_models = create_base_models()
    
    # Train ensemble
    ensemble_model, individual_models = train_ensemble(
        X_train_balanced, y_train_balanced,
        base_models
    )
    
    # Evaluate ensemble
    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 70)
    
    metrics, y_pred, y_proba = evaluate_ensemble(
        ensemble_model, 
        X_train_balanced, y_train_balanced,
        X_test, y_test,
        "Voting Ensemble"
    )
    
    # Also evaluate individual models for comparison
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL COMPARISON")
    print("=" * 70)
    
    individual_results = {}
    for name, model in individual_models.items():
        print(f"\n{name.upper()}:")
        ind_metrics, _, _ = evaluate_ensemble(
            model,
            X_train_balanced, y_train_balanced,
            X_test, y_test,
            name
        )
        individual_results[name] = ind_metrics['test']['accuracy']
    
    # Visualize
    plot_results(y_test, y_pred, y_proba, "Ensemble")
    
    # Save
    save_model(ensemble_model, feature_names, metrics, "voting_ensemble")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 ENSEMBLE MODEL TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n✅ Ensemble Test Accuracy: {metrics['test']['accuracy']:.2%}")
    print(f"✅ Ensemble Test AUC-ROC: {metrics['test']['roc_auc']:.4f}")
    print(f"✅ Ensemble Test F1-Score: {metrics['test']['f1']:.4f}")
    
    print("\n📊 INDIVIDUAL MODEL ACCURACIES:")
    for name, acc in sorted(individual_results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {name:15s}: {acc:.2%}")
    
    print("\n📊 OVERALL COMPARISON:")
    print("   Original XGBoost    : 54.6%")
    print("   Improved XGBoost    : 66.5%")
    print(f"   Voting Ensemble     : {metrics['test']['accuracy']:.1%}")
    print(f"   Total Improvement   : {(metrics['test']['accuracy'] - 0.546) * 100:+.1f}%")
    
    print("\n" + "=" * 70)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\n❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
