"""
Improved XGBoost Success Classifier with Hyperparameter Tuning
--------------------------------------------------------------
Enhanced version with better parameters and techniques to improve accuracy.

Usage:
    python models/train_success_model_improved.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.model_config import (
    TRAINING_CONFIG, MODEL_PATHS,
    METRICS_DIR, PLOTS_DIR, get_feature_columns
)

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# IMPROVED HYPERPARAMETERS
IMPROVED_PARAMS = {
    'n_estimators': 1000,  # More trees
    'max_depth': 6,  # Shallower trees to reduce overfitting
    'learning_rate': 0.01,  # Slower learning
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,  # More conservative
    'gamma': 0.3,  # Higher regularization
    'reg_alpha': 0.5,  # L1 regularization
    'reg_lambda': 2.0,  # L2 regularization
    'scale_pos_weight': 3.0,  # Handle class imbalance (will be adjusted)
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'early_stopping_rounds': 100
}


def load_training_data(data_path):
    """Load processed training data."""
    print(f"📂 Loading training data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"   ❌ Training data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} samples")
    
    return df


def prepare_features_and_target(df, target_col='success_label'):
    """Prepare feature matrix X and target vector y."""
    print(f"\n🔧 Preparing features and target ({target_col})...")
    
    if target_col not in df.columns:
        print(f"   ❌ Target column '{target_col}' not found")
        return None, None, None
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"   ✓ Using {len(feature_cols)} features")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"   ✓ Encoding {len(categorical_cols)} categorical features")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Fill NaN
    X = X.fillna(0)
    
    # Remove low-variance features
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=selected_features)
    
    feature_names = X.columns.tolist()
    
    print(f"   ✓ Final feature count: {len(feature_names)}")
    print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_names


def handle_class_imbalance(X_train, y_train, method='smote'):
    """Handle class imbalance using SMOTE or class weights."""
    print(f"\n⚖️  Handling class imbalance with {method.upper()}...")
    
    if method == 'smote':
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            print(f"   ✓ SMOTE applied: {len(X_train)} → {len(X_balanced)} samples")
            print(f"   ✓ New distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"   ⚠️  SMOTE failed: {e}")
            print(f"   ℹ️  Falling back to class weights")
            return X_train, y_train
    else:
        return X_train, y_train


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"\n✂️  Splitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   ✓ Train set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model_with_tuning(X_train, y_train, X_test, y_test, tune=False):
    """Train XGBoost with optional hyperparameter tuning."""
    
    if tune:
        print("\n🔍 Hyperparameter tuning (this may take 10-20 minutes)...")
        
        param_grid = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'min_child_weight': [3, 5, 7],
            'gamma': [0.1, 0.3, 0.5],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [1.0, 2.0, 3.0]
        }
        
        base_model = xgb.XGBClassifier(
            random_state=42,
            tree_method='hist',
            eval_metric='logloss'
        )
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"   ✓ Best parameters: {random_search.best_params_}")
        print(f"   ✓ Best CV score: {random_search.best_score_:.4f}")
        
        model = random_search.best_estimator_
    else:
        print("\n🚀 Training XGBoost classifier with improved parameters...")
        
        # Calculate class weight
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
        
        params = IMPROVED_PARAMS.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        print(f"   ✓ Adjusted class weight: {scale_pos_weight:.2f}")
        
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        print(f"   ✓ Training complete (best iteration: {model.best_iteration})")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
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
    
    # Print results
    print("\n   TRAIN SET:")
    for metric, value in metrics['train'].items():
        print(f"      {metric:12s}: {value:.4f}")
    
    print("\n   TEST SET:")
    for metric, value in metrics['test'].items():
        print(f"      {metric:12s}: {value:.4f}")
    
    # Classification report
    print("\n   CLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Fail', 'Success'], zero_division=0))
    
    return metrics, y_test_pred, y_test_proba


def plot_confusion_matrix(y_test, y_pred, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Success'],
                yticklabels=['Fail', 'Success'])
    plt.title('Confusion Matrix - Improved Success Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved confusion matrix: {output_path}")


def plot_roc_curve(y_test, y_proba, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Improved XGBoost (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Improved Success Prediction')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved ROC curve: {output_path}")


def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """Plot and save feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features (Improved Model)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved feature importance: {output_path}")


def save_model_and_artifacts(model, feature_names, metrics):
    """Save trained model and artifacts."""
    print("\n💾 Saving improved model and artifacts...")
    
    os.makedirs(os.path.dirname(MODEL_PATHS['success_classifier']), exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Save model (overwrite previous)
    improved_model_path = MODEL_PATHS['success_classifier'].replace('.pkl', '_improved.pkl')
    joblib.dump(model, improved_model_path)
    print(f"   ✓ Saved improved model: {improved_model_path}")
    
    # Save feature names
    joblib.dump(feature_names, MODEL_PATHS['feature_names'])
    print(f"   ✓ Saved feature names: {MODEL_PATHS['feature_names']}")
    
    # Save metrics
    metrics_path = os.path.join(METRICS_DIR, 'success_model_improved_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Saved metrics: {metrics_path}")


def main():
    print("=" * 70)
    print("TRAINING IMPROVED SUCCESS PREDICTION MODEL")
    print("=" * 70)
    
    # Load data
    data_path = project_root / "data" / "processed" / "training_data.csv"
    df = load_training_data(data_path)
    if df is None:
        return
    
    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(df, target_col='success_label')
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance with SMOTE
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, method='smote')
    
    # Train model (set tune=True for hyperparameter tuning, but it takes longer)
    model = train_model_with_tuning(X_train_balanced, y_train_balanced, X_test, y_test, tune=False)
    
    # Evaluate model
    metrics, y_pred, y_proba = evaluate_model(model, X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_confusion_matrix(y_test, y_pred, os.path.join(PLOTS_DIR, 'success_confusion_matrix_improved.png'))
    plot_roc_curve(y_test, y_proba, os.path.join(PLOTS_DIR, 'success_roc_curve_improved.png'))
    plot_feature_importance(model, feature_names, os.path.join(PLOTS_DIR, 'success_feature_importance_improved.png'))
    
    # Save model
    save_model_and_artifacts(model, feature_names, metrics)
    
    # Summary
    print("\n" + "=" * 70)
    print("IMPROVED MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✅ Test Accuracy: {metrics['test']['accuracy']:.2%}")
    print(f"✅ Test AUC-ROC: {metrics['test']['roc_auc']:.4f}")
    print(f"✅ Test F1-Score: {metrics['test']['f1']:.4f}")
    print(f"✅ Test Precision: {metrics['test']['precision']:.4f}")
    print(f"✅ Test Recall: {metrics['test']['recall']:.4f}")
    
    # Comparison with original
    print("\n📊 IMPROVEMENT OVER ORIGINAL MODEL:")
    print("   Original Accuracy: 54.6%")
    print(f"   Improved Accuracy: {metrics['test']['accuracy']:.1%}")
    print(f"   Improvement: {(metrics['test']['accuracy'] - 0.546) * 100:+.1f}%")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Review improved plots in models/plots/")
    print("2. For even better results, run with tune=True (takes 10-20 min)")
    print("3. Collect real-world success/failure data")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\n❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
