"""
Train XGBoost Success Classifier
---------------------------------
Train binary classification model to predict store success (success_label).

Usage:
    python models/train_success_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.model_config import (
    XGBOOST_CLASSIFIER_PARAMS, TRAINING_CONFIG, MODEL_PATHS,
    METRICS_DIR, PLOTS_DIR, get_feature_columns, calculate_class_weights
)

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_training_data(data_path):
    """Load processed training data."""
    print(f"📂 Loading training data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"   ❌ Training data not found: {data_path}")
        print("   ℹ️  Run: python scripts/prepare_training_data.py")
        return None
    
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} samples")
    
    return df


def prepare_features_and_target(df, target_col='success_label'):
    """
    Prepare feature matrix X and target vector y.
    
    Args:
        df: Training DataFrame
        target_col: Name of target column
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print(f"\n🔧 Preparing features and target ({target_col})...")
    
    # Check if target exists
    if target_col not in df.columns:
        print(f"   ❌ Target column '{target_col}' not found")
        return None, None, None
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"   ✓ Using {len(feature_cols)} features")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle categorical features with one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"   ✓ Encoding {len(categorical_cols)} categorical features")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    feature_names = X.columns.tolist()
    
    print(f"   ✓ Final feature count: {len(feature_names)}")
    print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"\n✂️  Splitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   ✓ Train set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, params):
    """
    Train XGBoost classifier with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Validation data
        params: Model hyperparameters
    
    Returns:
        Trained model
    """
    print("\n🚀 Training XGBoost classifier...")
    
    # Adjust class weights if needed
    if TRAINING_CONFIG['handle_imbalance']:
        scale_pos_weight = calculate_class_weights(y_train)
        params = params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        print(f"   ✓ Adjusted class weight: {scale_pos_weight:.2f}")
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    print(f"   ✓ Training complete (best iteration: {model.best_iteration})")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Returns:
        Dictionary with evaluation metrics
    """
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
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
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
    print(classification_report(y_test, y_test_pred, target_names=['Fail', 'Success']))
    
    return metrics, y_test_pred, y_test_proba


def plot_confusion_matrix(y_test, y_pred, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Success'],
                yticklabels=['Fail', 'Success'])
    plt.title('Confusion Matrix - Success Prediction')
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
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Success Prediction')
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
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved feature importance: {output_path}")


def save_model_and_artifacts(model, feature_names, metrics):
    """Save trained model, feature names, and metrics."""
    print("\n💾 Saving model and artifacts...")
    
    # Create directories
    os.makedirs(os.path.dirname(MODEL_PATHS['success_classifier']), exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATHS['success_classifier'])
    print(f"   ✓ Saved model: {MODEL_PATHS['success_classifier']}")
    
    # Save feature names
    joblib.dump(feature_names, MODEL_PATHS['feature_names'])
    print(f"   ✓ Saved feature names: {MODEL_PATHS['feature_names']}")
    
    # Save metrics
    metrics_path = os.path.join(METRICS_DIR, 'success_model_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Saved metrics: {metrics_path}")


def main():
    print("=" * 70)
    print("TRAINING SUCCESS PREDICTION MODEL (XGBoost Classifier)")
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
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size=TRAINING_CONFIG['test_size'],
        random_state=TRAINING_CONFIG['random_state']
    )
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, XGBOOST_CLASSIFIER_PARAMS)
    
    # Evaluate model
    metrics, y_pred, y_proba = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_confusion_matrix(y_test, y_pred, os.path.join(PLOTS_DIR, 'success_confusion_matrix.png'))
    plot_roc_curve(y_test, y_proba, os.path.join(PLOTS_DIR, 'success_roc_curve.png'))
    plot_feature_importance(model, feature_names, os.path.join(PLOTS_DIR, 'success_feature_importance.png'))
    
    # Save model
    save_model_and_artifacts(model, feature_names, metrics)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✅ Test Accuracy: {metrics['test']['accuracy']:.2%}")
    print(f"✅ Test AUC-ROC: {metrics['test']['roc_auc']:.4f}")
    print(f"✅ Test F1-Score: {metrics['test']['f1']:.4f}")
    print("\n📁 Model saved to:", MODEL_PATHS['success_classifier'])
    print("📁 Plots saved to:", PLOTS_DIR)
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Review feature importance plot")
    print("2. Train revenue model: python models/train_revenue_model.py")
    print("3. Make predictions: python scripts/predict_locations.py")
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
