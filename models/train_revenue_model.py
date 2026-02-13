"""
Train XGBoost Revenue Regressor
--------------------------------
Train regression model to predict monthly revenue.

Usage:
    python models/train_revenue_model.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.model_config import (
    XGBOOST_REGRESSOR_PARAMS, TRAINING_CONFIG, MODEL_PATHS,
    METRICS_DIR, PLOTS_DIR, get_feature_columns
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


def prepare_features_and_target(df, target_col='monthly_revenue'):
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
    
    feature_names = X.columns.tolist()
    
    print(f"   ✓ Final feature count: {len(feature_names)}")
    print(f"   ✓ Target stats: mean=₹{y.mean():,.0f}, median=₹{y.median():,.0f}, std=₹{y.std():,.0f}")
    
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"\n✂️  Splitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"   ✓ Train set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, params):
    """Train XGBoost regressor with early stopping."""
    print("\n🚀 Training XGBoost regressor...")
    
    model = xgb.XGBRegressor(**params)
    
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
    
    # Calculate metrics
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'mape': np.mean(np.abs((y_train - y_train_pred) / (y_train + 1))) * 100
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'mape': np.mean(np.abs((y_test - y_test_pred) / (y_test + 1))) * 100
        }
    }
    
    # Print results
    print("\n   TRAIN SET:")
    print(f"      R² Score    : {metrics['train']['r2']:.4f}")
    print(f"      RMSE        : ₹{metrics['train']['rmse']:,.0f}")
    print(f"      MAE         : ₹{metrics['train']['mae']:,.0f}")
    print(f"      MAPE        : {metrics['train']['mape']:.2f}%")
    
    print("\n   TEST SET:")
    print(f"      R² Score    : {metrics['test']['r2']:.4f}")
    print(f"      RMSE        : ₹{metrics['test']['rmse']:,.0f}")
    print(f"      MAE         : ₹{metrics['test']['mae']:,.0f}")
    print(f"      MAPE        : {metrics['test']['mape']:.2f}%")
    
    return metrics, y_test_pred


def plot_predictions(y_test, y_pred, output_path):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Monthly Revenue (₹)')
    plt.ylabel('Predicted Monthly Revenue (₹)')
    plt.title('Revenue Prediction: Actual vs Predicted')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved prediction plot: {output_path}")


def plot_residuals(y_test, y_pred, output_path):
    """Plot residuals distribution."""
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals scatter
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Monthly Revenue (₹)')
    axes[0].set_ylabel('Residuals (₹)')
    axes[0].set_title('Residual Plot')
    axes[0].grid(alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].set_xlabel('Residuals (₹)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved residuals plot: {output_path}")


def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features (Revenue Prediction)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved feature importance: {output_path}")


def save_model_and_artifacts(model, feature_names, metrics):
    """Save trained model and artifacts."""
    print("\n💾 Saving model and artifacts...")
    
    os.makedirs(os.path.dirname(MODEL_PATHS['revenue_regressor']), exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATHS['revenue_regressor'])
    print(f"   ✓ Saved model: {MODEL_PATHS['revenue_regressor']}")
    
    # Save metrics
    metrics_path = os.path.join(METRICS_DIR, 'revenue_model_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Saved metrics: {metrics_path}")


def main():
    print("=" * 70)
    print("TRAINING REVENUE PREDICTION MODEL (XGBoost Regressor)")
    print("=" * 70)
    
    # Load data
    data_path = project_root / "data" / "processed" / "training_data.csv"
    df = load_training_data(data_path)
    if df is None:
        return
    
    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(df, target_col='monthly_revenue')
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=TRAINING_CONFIG['test_size'],
        random_state=TRAINING_CONFIG['random_state']
    )
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, XGBOOST_REGRESSOR_PARAMS)
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_predictions(y_test, y_pred, os.path.join(PLOTS_DIR, 'revenue_predictions.png'))
    plot_residuals(y_test, y_pred, os.path.join(PLOTS_DIR, 'revenue_residuals.png'))
    plot_feature_importance(model, feature_names, os.path.join(PLOTS_DIR, 'revenue_feature_importance.png'))
    
    # Save model
    save_model_and_artifacts(model, feature_names, metrics)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✅ Test R² Score: {metrics['test']['r2']:.4f}")
    print(f"✅ Test RMSE: ₹{metrics['test']['rmse']:,.0f}")
    print(f"✅ Test MAE: ₹{metrics['test']['mae']:,.0f}")
    print("\n📁 Model saved to:", MODEL_PATHS['revenue_regressor'])
    print("📁 Plots saved to:", PLOTS_DIR)
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Review prediction plots and feature importance")
    print("2. Make predictions: python scripts/predict_locations.py")
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
