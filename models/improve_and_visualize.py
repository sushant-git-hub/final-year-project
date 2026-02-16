"""
Improve category-specific models through hyperparameter tuning and create visualization plots
"""
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
                             classification_report)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

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
        'rating', 'location_score',
        'daily_footfall', 'monthly_revenue'
    ]
    
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    y = df['success_label']
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    feature_names = X.columns.tolist()
    return X, y, feature_names

def hyperparameter_tuning(X_train, y_train, category):
    """Perform grid search for hyperparameter tuning"""
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING FOR {category.upper()}")
    print(f"{'='*70}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [600, 800, 1000],
        'max_depth': [5, 6, 7, 8],
        'learning_rate': [0.02, 0.03, 0.05],
        'min_child_weight': [3, 5, 7],
        'subsample': [0.8, 0.85, 0.9],
        'colsample_bytree': [0.8, 0.85, 0.9],
        'gamma': [0.1, 0.15, 0.2],
        'reg_alpha': [1.0, 1.5, 2.0],
        'reg_lambda': [1.5, 2.0, 2.5]
    }
    
    # Create base model
    base_model = xgb.XGBClassifier(
        random_state=42,
        tree_method='hist',
        eval_metric='logloss'
    )
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nRunning grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def test_sampling_strategies(X_train, y_train, X_test, y_test, params):
    """Test different oversampling strategies"""
    print(f"\n{'='*70}")
    print("TESTING SAMPLING STRATEGIES")
    print(f"{'='*70}")
    
    strategies = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
        'None': None
    }
    
    results = {}
    
    for name, sampler in strategies.items():
        print(f"\nTesting {name}...")
        
        if sampler:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_resampled, y_resampled)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {'accuracy': acc, 'f1': f1}
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ Best strategy: {best_strategy[0]} (Accuracy: {best_strategy[1]['accuracy']:.4f})")
    
    return best_strategy[0], strategies[best_strategy[0]]

def create_visualizations(model, X_test, y_test, feature_names, category, metrics):
    """Create comprehensive visualization plots"""
    print(f"\n{'='*70}")
    print(f"CREATING VISUALIZATIONS FOR {category.upper()}")
    print(f"{'='*70}")
    
    # Create plots directory
    plots_dir = Path(f'models/plots/{category}')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance Plot
    print("\n1. Creating feature importance plot...")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(20), importance[indices], color='steelblue')
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top 20 Feature Importance - {category.upper()}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plots_dir / 'feature_importance.png'}")
    
    # 2. Confusion Matrix
    print("2. Creating confusion matrix...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Failure', 'Success'],
                yticklabels=['Failure', 'Success'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {category.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plots_dir / 'confusion_matrix.png'}")
    
    # 3. ROC Curve
    print("3. Creating ROC curve...")
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {category.upper()}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plots_dir / 'roc_curve.png'}")
    
    # 4. Precision-Recall Curve
    print("4. Creating precision-recall curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {category.upper()}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plots_dir / 'precision_recall_curve.png'}")
    
    # 5. Metrics Summary Plot
    print("5. Creating metrics summary plot...")
    metrics_data = {
        'Accuracy': metrics['test_accuracy'],
        'Precision': metrics['test_precision'],
        'Recall': metrics['test_recall'],
        'F1-Score': metrics['test_f1'],
        'AUC': metrics['test_auc']
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_data.keys(), metrics_data.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylim([0, 1])
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Model Performance Metrics - {category.upper()}', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plots_dir / 'metrics_summary.png'}")
    
    print(f"\n✓ All plots saved to: {plots_dir}")

def train_improved_model(category):
    """Train improved model with tuning and visualizations"""
    print(f"\n{'='*70}")
    print(f"TRAINING IMPROVED MODEL FOR {category.upper()}")
    print(f"{'='*70}")
    
    # Load data
    X, y, feature_names = load_category_data(category)
    print(f"\nDataset: {len(X)} records, {len(feature_names)} features")
    print(f"Success rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(X_train, y_train, category)
    
    # Test sampling strategies
    best_strategy_name, best_sampler = test_sampling_strategies(
        X_train, y_train, X_test, y_test, best_params
    )
    
    # Train final model with best configuration
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}")
    
    if best_sampler:
        X_train_final, y_train_final = best_sampler.fit_resample(X_train, y_train)
        print(f"\nApplied {best_strategy_name}")
        print(f"Training samples after resampling: {len(X_train_final)}")
    else:
        X_train_final, y_train_final = X_train, y_train
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_final, y_train_final)
    
    # Evaluate
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n--- Final Model Performance ---")
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc:.1%})")
    print(f"Test Accuracy    : {test_acc:.4f} ({test_acc:.1%})")
    print(f"Test Precision   : {test_precision:.4f}")
    print(f"Test Recall      : {test_recall:.4f}")
    print(f"Test F1-Score    : {test_f1:.4f}")
    print(f"Test AUC         : {test_auc:.4f}")
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(final_model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model
    save_dir = Path('models/category_specific/improved')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f'{category}_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"\n✓ Saved model to: {model_path}")
    
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
        'best_params': best_params,
        'sampling_strategy': best_strategy_name,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    metrics_path = save_dir / f'{category}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    # Create visualizations
    create_visualizations(final_model, X_test, y_test, feature_names, category, metrics)
    
    return final_model, metrics

def create_comparison_plots(all_metrics):
    """Create comparison plots across all categories"""
    print(f"\n{'='*70}")
    print("CREATING CATEGORY COMPARISON PLOTS")
    print(f"{'='*70}")
    
    plots_dir = Path('models/plots/comparison')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    categories = list(all_metrics.keys())
    accuracies = [all_metrics[cat]['test_accuracy'] for cat in categories]
    f1_scores = [all_metrics[cat]['test_f1'] for cat in categories]
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, accuracies, color='steelblue', alpha=0.8)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison Across Categories', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plots_dir / 'accuracy_comparison.png'}")
    
    # 2. Multi-metric Comparison
    metrics_df = pd.DataFrame({
        'Category': categories,
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'Precision': [all_metrics[cat]['test_precision'] for cat in categories],
        'Recall': [all_metrics[cat]['test_recall'] for cat in categories]
    })
    
    metrics_df_melted = metrics_df.melt(id_vars='Category', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=metrics_df_melted, x='Category', y='Score', hue='Metric')
    plt.ylabel('Score', fontsize=12)
    plt.title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.legend(title='Metric', loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'multi_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plots_dir / 'multi_metric_comparison.png'}")
    
    print(f"\n✓ Comparison plots saved to: {plots_dir}")

def main():
    print("="*70)
    print("MODEL IMPROVEMENT AND VISUALIZATION")
    print("="*70)
    
    # Categories to improve
    categories = ['food', 'retail_general', 'retail_fashion', 'retail_electronics', 'services']
    
    all_metrics = {}
    
    for category in categories:
        try:
            model, metrics = train_improved_model(category)
            all_metrics[category] = metrics
        except Exception as e:
            print(f"\n❌ Error improving {category} model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison plots
    if all_metrics:
        create_comparison_plots(all_metrics)
    
    # Summary
    print(f"\n{'='*70}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*70}")
    
    for category, metrics in all_metrics.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"  F1-Score     : {metrics['test_f1']:.4f}")
        print(f"  CV Accuracy  : {metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
        print(f"  Best Strategy: {metrics['sampling_strategy']}")
    
    print(f"\n{'='*70}")
    print("✓ MODEL IMPROVEMENT COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
