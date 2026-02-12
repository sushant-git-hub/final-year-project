"""
Systematic data leakage detection script
Identifies which features are causing 100% accuracy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load retail training data"""
    df = pd.read_csv('data/processed/category_specific/training_data_retail.csv')
    
    # Columns to drop
    cols_to_drop = [
        'place_id', 'name', 'success_label', 'main_category',
        'cell_id', 'ward_id', 'ward_name', 'locality',
        'footfall_category', 'profitability', 'rental_zone',
        'ward_zone', 'income_tier', 'tier', 'confidence'
    ]
    
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    y = df['success_label']
    
    # Handle any remaining categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def test_single_feature(X, y, feature_name):
    """Test if a single feature can predict target perfectly"""
    X_single = X[[feature_name]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_single, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try simple logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    acc = accuracy_score(y_test, lr.predict(X_test))
    
    return acc

def test_feature_combinations(X, y):
    """Test different feature combinations to find leakage"""
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL FEATURES FOR DATA LEAKAGE")
    print("="*70)
    
    feature_scores = {}
    
    for feature in X.columns:
        try:
            acc = test_single_feature(X, y, feature)
            feature_scores[feature] = acc
            
            if acc > 0.95:
                print(f"🚨 LEAKAGE DETECTED: {feature:40s} → {acc:.2%} accuracy")
            elif acc > 0.85:
                print(f"⚠️  SUSPICIOUS:      {feature:40s} → {acc:.2%} accuracy")
        except Exception as e:
            print(f"❌ ERROR:           {feature:40s} → {str(e)[:30]}")
    
    # Sort by accuracy
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 10 MOST PREDICTIVE FEATURES")
    print("="*70)
    for feature, acc in sorted_features[:10]:
        print(f"  {feature:40s}: {acc:.2%}")
    
    return sorted_features

def test_without_suspicious_features(X, y, suspicious_features):
    """Test model performance after removing suspicious features"""
    print("\n" + "="*70)
    print("TESTING WITHOUT SUSPICIOUS FEATURES")
    print("="*70)
    
    # Remove suspicious features
    X_clean = X.drop(suspicious_features, axis=1, errors='ignore')
    
    print(f"\nOriginal features: {len(X.columns)}")
    print(f"Removed features: {len(suspicious_features)}")
    print(f"Remaining features: {len(X_clean.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test with different models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    print("\n--- Model Performance ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        print(f"{name:20s}: Train={train_acc:.2%}, Test={test_acc:.2%}")
        
        if test_acc < 0.95:
            print(f"  ✅ Realistic accuracy achieved!")
        else:
            print(f"  ⚠️  Still suspiciously high")

def analyze_feature_correlations(X, y):
    """Analyze correlations between features and target"""
    print("\n" + "="*70)
    print("FEATURE-TARGET CORRELATIONS")
    print("="*70)
    
    correlations = {}
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            corr = X[col].corr(y)
            correlations[col] = abs(corr)
    
    # Sort by correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 features by correlation with target:")
    for feature, corr in sorted_corr[:15]:
        if corr > 0.7:
            print(f"🚨 {feature:40s}: {corr:.3f} (VERY HIGH)")
        elif corr > 0.5:
            print(f"⚠️  {feature:40s}: {corr:.3f} (HIGH)")
        else:
            print(f"   {feature:40s}: {corr:.3f}")
    
    return sorted_corr

def check_for_duplicates(X, y):
    """Check if there are duplicate rows that could cause overfitting"""
    print("\n" + "="*70)
    print("CHECKING FOR DUPLICATE ROWS")
    print("="*70)
    
    # Check for exact duplicates
    duplicates = X.duplicated().sum()
    print(f"Exact duplicate rows: {duplicates}")
    
    if duplicates > 0:
        print("⚠️  Found duplicate rows - this could cause data leakage!")
        
        # Check if duplicates have same target
        df_with_target = X.copy()
        df_with_target['target'] = y
        
        duplicate_mask = df_with_target.duplicated(subset=X.columns, keep=False)
        duplicate_rows = df_with_target[duplicate_mask]
        
        print(f"\nDuplicate rows with target distribution:")
        print(duplicate_rows.groupby(X.columns.tolist())['target'].value_counts().head(10))

def main():
    print("="*70)
    print("DATA LEAKAGE DETECTION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X, y = load_data()
    print(f"Dataset: {len(X)} records, {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # 1. Check for duplicates
    check_for_duplicates(X, y)
    
    # 2. Analyze correlations
    correlations = analyze_feature_correlations(X, y)
    
    # 3. Test individual features
    feature_scores = test_feature_combinations(X, y)
    
    # 4. Identify suspicious features (>85% accuracy alone)
    suspicious_features = [f for f, acc in feature_scores if acc > 0.85]
    
    if suspicious_features:
        print(f"\n{'='*70}")
        print(f"IDENTIFIED {len(suspicious_features)} SUSPICIOUS FEATURES:")
        print(f"{'='*70}")
        for feature in suspicious_features:
            acc = dict(feature_scores)[feature]
            print(f"  - {feature:40s} ({acc:.2%})")
        
        # 5. Test without suspicious features
        test_without_suspicious_features(X, y, suspicious_features)
    else:
        print("\n✅ No individual features show data leakage")
        print("   The 100% accuracy may be due to feature combinations")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if suspicious_features:
        print("\n1. Remove the following features from training:")
        for feature in suspicious_features[:5]:
            print(f"   - {feature}")
        
        print("\n2. Re-train models without these features")
        print("\n3. Expected accuracy should drop to realistic levels (70-75%)")
    else:
        print("\n1. Investigate feature combinations")
        print("2. Try simpler models (logistic regression)")
        print("3. Use more aggressive regularization")

if __name__ == "__main__":
    main()
