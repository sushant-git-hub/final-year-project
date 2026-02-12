"""
Batch Prediction Script
-----------------------
Generate predictions for all grid cells and save to PostGIS.

Usage:
    python scripts/predict_locations.py
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.project.feature_engineering import (
    add_city_center_distance,
    add_interaction_features,
    handle_missing_values
)
from models.model_config import MODEL_PATHS, get_feature_columns

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_env():
    """Load database configuration."""
    load_dotenv()
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "db": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "port": os.getenv("DB_PORT", "5432"),
    }


def get_engine(cfg):
    """Create SQLAlchemy engine."""
    url = (
        f"postgresql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['db']}"
    )
    return create_engine(url)


def load_models():
    """Load trained models."""
    print("📂 Loading trained models...")
    
    models = {}
    
    # Load success classifier
    if os.path.exists(MODEL_PATHS['success_classifier']):
        models['success'] = joblib.load(MODEL_PATHS['success_classifier'])
        print(f"   ✓ Loaded success classifier")
    else:
        print(f"   ⚠️  Success classifier not found: {MODEL_PATHS['success_classifier']}")
    
    # Load revenue regressor
    if os.path.exists(MODEL_PATHS['revenue_regressor']):
        models['revenue'] = joblib.load(MODEL_PATHS['revenue_regressor'])
        print(f"   ✓ Loaded revenue regressor")
    else:
        print(f"   ⚠️  Revenue regressor not found: {MODEL_PATHS['revenue_regressor']}")
    
    # Load feature names
    if os.path.exists(MODEL_PATHS['feature_names']):
        models['feature_names'] = joblib.load(MODEL_PATHS['feature_names'])
        print(f"   ✓ Loaded feature names ({len(models['feature_names'])} features)")
    else:
        print(f"   ⚠️  Feature names not found: {MODEL_PATHS['feature_names']}")
    
    if not models:
        print("   ❌ No models found. Please train models first.")
        return None
    
    return models


def load_grid_features(engine):
    """Load all grid cell features from PostGIS."""
    print("\n📂 Loading grid features from PostGIS...")
    
    query = """
    SELECT 
        g.cell_id,
        g.center_lat,
        g.center_lon,
        ST_AsText(g.geometry) as geometry_wkt,
        p.*, r.*, rf.*, d.*, f.*, t.*, i.*
    FROM grid_cells g
    LEFT JOIN poi_features p ON g.cell_id = p.cell_id
    LEFT JOIN road_features r ON g.cell_id = r.cell_id
    LEFT JOIN rental_features rf ON g.cell_id = rf.cell_id
    LEFT JOIN demographic_features d ON g.cell_id = d.cell_id
    LEFT JOIN footfall_features f ON g.cell_id = f.cell_id
    LEFT JOIN transit_features t ON g.cell_id = t.cell_id
    LEFT JOIN income_features i ON g.cell_id = i.cell_id
    """
    
    try:
        df = pd.read_sql(query, engine)
        print(f"   ✓ Loaded {len(df)} grid cells")
        return df
    except Exception as e:
        print(f"   ❌ Error loading features: {e}")
        return None


def prepare_features_for_prediction(df, feature_names):
    """Prepare features for prediction."""
    print("\n🔧 Preparing features for prediction...")
    
    # Apply feature engineering
    df = add_city_center_distance(df)
    df = add_interaction_features(df)
    df = handle_missing_values(df, strategy='median')
    
    # Get feature columns (excluding metadata)
    available_cols = df.columns.tolist()
    
    # Handle categorical encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols = ['cell_id', 'geometry_wkt']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align features with training features
    X = pd.DataFrame()
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0  # Missing features default to 0
    
    print(f"   ✓ Prepared {len(X)} samples with {len(feature_names)} features")
    
    return X


def make_predictions(models, X, df):
    """Generate predictions using trained models."""
    print("\n🔮 Generating predictions...")
    
    predictions = df[['cell_id', 'center_lat', 'center_lon']].copy()
    
    # Success probability
    if 'success' in models:
        success_proba = models['success'].predict_proba(X)[:, 1]
        predictions['success_probability'] = success_proba
        predictions['predicted_success'] = (success_proba >= 0.5).astype(int)
        print(f"   ✓ Generated success predictions (avg probability: {success_proba.mean():.2%})")
    
    # Revenue prediction
    if 'revenue' in models:
        revenue_pred = models['revenue'].predict(X)
        predictions['predicted_monthly_revenue'] = revenue_pred.clip(min=0)  # No negative revenue
        print(f"   ✓ Generated revenue predictions (avg: ₹{revenue_pred.mean():,.0f})")
    
    # Location score (0-100)
    if 'success_probability' in predictions.columns:
        predictions['location_score'] = (predictions['success_probability'] * 100).round(2)
    
    # Recommendation category
    if 'success_probability' in predictions.columns:
        def categorize_location(prob):
            if prob >= 0.75:
                return 'Excellent'
            elif prob >= 0.60:
                return 'Good'
            elif prob >= 0.45:
                return 'Fair'
            else:
                return 'Poor'
        
        predictions['recommendation'] = predictions['success_probability'].apply(categorize_location)
    
    return predictions


def save_predictions_to_postgis(predictions, engine):
    """Save predictions to PostGIS table."""
    print("\n💾 Saving predictions to PostGIS...")
    
    # Create GeoDataFrame
    predictions_gdf = gpd.GeoDataFrame(
        predictions,
        geometry=gpd.points_from_xy(predictions.center_lon, predictions.center_lat),
        crs='EPSG:4326'
    )
    
    # Write to PostGIS
    try:
        predictions_gdf.to_postgis(
            'location_predictions',
            engine,
            if_exists='replace',
            index=False
        )
        print(f"   ✓ Saved {len(predictions)} predictions to 'location_predictions' table")
    except Exception as e:
        print(f"   ❌ Error saving to PostGIS: {e}")
        
        # Fallback: save to CSV
        csv_path = project_root / "data" / "processed" / "location_predictions.csv"
        predictions.to_csv(csv_path, index=False)
        print(f"   ✓ Saved predictions to CSV: {csv_path}")


def print_summary(predictions):
    """Print prediction summary statistics."""
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    
    if 'success_probability' in predictions.columns:
        print(f"\nSuccess Probability:")
        print(f"  • Mean: {predictions['success_probability'].mean():.2%}")
        print(f"  • Median: {predictions['success_probability'].median():.2%}")
        print(f"  • Min: {predictions['success_probability'].min():.2%}")
        print(f"  • Max: {predictions['success_probability'].max():.2%}")
    
    if 'predicted_monthly_revenue' in predictions.columns:
        print(f"\nPredicted Monthly Revenue:")
        print(f"  • Mean: ₹{predictions['predicted_monthly_revenue'].mean():,.0f}")
        print(f"  • Median: ₹{predictions['predicted_monthly_revenue'].median():,.0f}")
        print(f"  • Min: ₹{predictions['predicted_monthly_revenue'].min():,.0f}")
        print(f"  • Max: ₹{predictions['predicted_monthly_revenue'].max():,.0f}")
    
    if 'recommendation' in predictions.columns:
        print(f"\nRecommendation Distribution:")
        print(predictions['recommendation'].value_counts().to_string())


def main():
    print("=" * 70)
    print("BATCH PREDICTION FOR ALL GRID CELLS")
    print("=" * 70)
    
    # Load models
    models = load_models()
    if models is None:
        print("\n❌ Please train models first:")
        print("   1. python models/train_success_model.py")
        print("   2. python models/train_revenue_model.py")
        return
    
    # Load database config
    cfg = load_env()
    engine = get_engine(cfg)
    
    # Load grid features
    df = load_grid_features(engine)
    if df is None:
        return
    
    # Prepare features
    X = prepare_features_for_prediction(df, models['feature_names'])
    
    # Make predictions
    predictions = make_predictions(models, X, df)
    
    # Save to PostGIS
    save_predictions_to_postgis(predictions, engine)
    
    # Print summary
    print_summary(predictions)
    
    print("\n" + "=" * 70)
    print("✅ PREDICTIONS COMPLETE!")
    print("=" * 70)
    print("\nQuery predictions in PostgreSQL:")
    print("  SELECT * FROM location_predictions")
    print("  WHERE success_probability > 0.7")
    print("  ORDER BY predicted_monthly_revenue DESC")
    print("  LIMIT 10;")
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
