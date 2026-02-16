"""
Quick model improvement with better hyperparameters and comprehensive visualizations
"""
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Improved hyperparameters based on best practices
IMPROVED_PARAMS = {
    'food': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.02,
        'min_child_weight': 5,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'gamma': 0.2,
        'reg_alpha': 1.5,
        'reg_lambda': 2.0,
        'scale_pos_weight': 1.1,
        'random_state': 42,
        'tree_method': 'hist'
    },
    'retail_general': {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.02,
        'min_child_weight': 3,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0.1,
        'reg_alpha': 1.0,
        'reg_lambda': 1.5,
        'scale_pos_weight': 1.2,
        'random_state': 42,
        'tree_method': 'hist'
    },
    'retail_fashion': {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.02,
        'min_child_weight': 3,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0.15,
        'reg_alpha': 1.2,
        'reg_lambda': 1.8,
        'scale_pos_weight': 1.15,
        'random_state': 42,
        'tree_method': 'hist'
    },
    'retail_electronics': {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.02,
        'min_child_weight': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'gamma': 0.15,
        'reg_alpha': 1.3,
        'reg_lambda': 1.9,
        'scale_pos_weight': 1.2,
        'random_state': 42,
        'tree_method': 'hist'
    },
    'services': {
        'n_estimators': 800,
        'max_depth': 5,
        'learning_rate': 0.03,
        'min_child_weight': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.25,
        'reg_alpha': 1.7,
        'reg_lambda': 2.2,
        'scale_pos_weight': 1.1,
        'random_state': 42,
        'tree_method': 'hist'
    }
}

def load_category_data(category):
    """Load category-specific training data"""
    filepath = f'data/processed/category_specific/training_data_{category}.csv'
    df = pd.read_csv(filepath)
    
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
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, X.columns.tolist()

def create_visualizations(model, X_test, y_test, feature_names, category, metrics):
    """Create comprehensive visualization plots"""
    plots_dir = Path(f'models/plots/{category}')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    plt.barh(range(20), importance[indices], color=colors)
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top 20 Features - {category.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                xticklabels=['Failure', 'Success'],
                yticklabels=['Failure', 'Success'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {category.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='orange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {category.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=3)
    plt.fill_between(recall, precision, alpha=0.2, color='green')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(f'Precision-Recall Curve - {category.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Metrics Summary
    metrics_data = {
        'Accuracy': metrics['test_accuracy'],
        'Precision': metrics['test_precision'],
        'Recall': metrics['test_recall'],
        'F1-Score': metrics['test_f1'],
        'AUC': metrics['test_auc']
    }
    
    plt.figure(figsize=(10, 6))
    colors_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = plt.bar(metrics_data.keys(), metrics_data.values(), color=colors_map, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylim([0, 1])
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Performance Metrics - {category.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Created 5 visualization plots in: {plots_dir}")

def train_improved_model(category):
    """Train improved model"""
    print(f"\n{'='*70}")
    print(f"TRAINING IMPROVED MODEL: {category.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    
    X, y, feature_names = load_category_data(category)
    print(f"Dataset: {len(X)} records, {len(feature_names)} features")
    print(f"Success rate: {y.mean():.2%}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_balanced)} samples")
    
    # Train with improved parameters
    print(f"\nTraining with optimized hyperparameters...")
    params = IMPROVED_PARAMS[category]
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n--- Performance ---")
    print(f"Train Accuracy: {train_acc:.4f} ({train_acc:.1%})")
    print(f"Test Accuracy : {test_acc:.4f} ({test_acc:.1%})")
    print(f"Precision     : {test_precision:.4f}")
    print(f"Recall        : {test_recall:.4f}")
    print(f"F1-Score      : {test_f1:.4f}")
    print(f"AUC           : {test_auc:.4f}")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save
    save_dir = Path('models/category_specific/improved')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, save_dir / f'{category}_model.pkl')
    joblib.dump(feature_names, save_dir / f'{category}_features.pkl')
    
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
        'hyperparameters': params,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    with open(save_dir / f'{category}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved model and metrics")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_visualizations(model, X_test, y_test, feature_names, category, metrics)
    
    return model, metrics

def create_comparison_plots(all_metrics):
    """Create comparison plots"""
    print(f"\n{'='*70}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*70}")
    
    plots_dir = Path('models/plots/comparison')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    categories = [cat.replace('_', ' ').title() for cat in all_metrics.keys()]
    accuracies = [m['test_accuracy'] for m in all_metrics.values()]
    f1_scores = [m['test_f1'] for m in all_metrics.values()]
    
    # Accuracy Comparison
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison Across Categories', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(accuracies):.1%}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Multi-metric Comparison
    metrics_df = pd.DataFrame({
        'Category': categories,
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'Precision': [m['test_precision'] for m in all_metrics.values()],
        'Recall': [m['test_recall'] for m in all_metrics.values()]
    })
    
    metrics_melted = metrics_df.melt(id_vars='Category', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=metrics_melted, x='Category', y='Score', hue='Metric', palette='Set2')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.legend(title='Metric', loc='lower right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'multi_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plots to: {plots_dir}")

def main():
    print("="*70)
    print("MODEL IMPROVEMENT & VISUALIZATION")
    print("="*70)
    
    categories = ['food', 'retail_general', 'retail_fashion', 'retail_electronics', 'services']
    all_metrics = {}
    
    for category in categories:
        try:
            model, metrics = train_improved_model(category)
            all_metrics[category] = metrics
        except Exception as e:
            print(f"\n❌ Error with {category}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_metrics:
        create_comparison_plots(all_metrics)
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    for category, metrics in all_metrics.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"  F1-Score: {metrics['test_f1']:.4f}")
        print(f"  CV Score: {metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
    
    avg_acc = np.mean([m['test_accuracy'] for m in all_metrics.values()])
    print(f"\n{'='*70}")
    print(f"AVERAGE ACCURACY: {avg_acc:.2%}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
