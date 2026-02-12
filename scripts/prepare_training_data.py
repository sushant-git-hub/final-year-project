"""
Prepare Training Data for ML Models
------------------------------------
Load features from PostGIS, join with training labels, and prepare for ML training.

Usage:
    python scripts/prepare_training_data.py
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.project.feature_engineering import (
    add_city_center_distance,
    add_interaction_features,
    handle_missing_values,
    get_feature_columns
)

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_env():
    """Load database configuration from .env file."""
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


def load_features_from_postgis(engine):
    """
    Load and join all feature tables from PostGIS.
    
    Returns:
        GeoDataFrame with all features joined by cell_id
    """
    print("📂 Loading features from PostGIS...")
    
    # Main query to join all feature tables
    query = """
    SELECT 
        g.cell_id,
        g.center_lat,
        g.center_lon,
        ST_AsText(g.geometry) as geometry_wkt,
        
        -- POI features
        p.competitor_count,
        p.nearest_m,
        
        -- Road features
        r.road_density_km,
        r.major_dist_m,
        
        -- Rental features
        rf.commercial_rent_per_sqft,
        rf.zone as rental_zone,
        rf.tier,
        rf.confidence,
        rf.locality,
        rf.locality_dist_m,
        
        -- Demographic features
        d.total_population,
        d.ward_id,
        d.ward_name,
        d.zone as ward_zone,
        d.ward_dist_m,
        
        -- Footfall features
        f.footfall_generator_count,
        f.nearest_generator_m,
        
        -- Transit features
        t.transit_stop_count,
        t.nearest_transit_m,
        
        -- Income features
        i.avg_monthly_income,
        i.property_price_sqft,
        i.purchasing_power_index,
        i.income_tier
        
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
        features_df = pd.read_sql(query, engine)
        print(f"   ✓ Loaded {len(features_df)} grid cells with features")
        print(f"   ✓ Feature columns: {len(features_df.columns)}")
        return features_df
    except Exception as e:
        print(f"   ❌ Error loading features: {e}")
        print("   ℹ️  Make sure feature_pipeline.py has been run to populate tables")
        return None


def load_training_labels(labels_path):
    """
    Load training labels from CSV.
    
    Returns:
        DataFrame with training labels
    """
    print(f"\n📂 Loading training labels from {labels_path}...")
    
    if not os.path.exists(labels_path):
        print(f"   ❌ Labels file not found: {labels_path}")
        return None
    
    labels_df = pd.read_csv(labels_path)
    print(f"   ✓ Loaded {len(labels_df)} labeled stores")
    
    return labels_df


def spatial_join_labels_to_grid(labels_df, features_df):
    """
    Perform spatial join to assign each store to its nearest grid cell.
    
    Args:
        labels_df: DataFrame with store labels (must have latitude, longitude)
        features_df: DataFrame with grid features (must have center_lat, center_lon)
    
    Returns:
        DataFrame with labels joined to grid cells
    """
    print("\n🗺️  Performing spatial join...")
    
    # Create GeoDataFrames
    stores_gdf = gpd.GeoDataFrame(
        labels_df,
        geometry=gpd.points_from_xy(labels_df.longitude, labels_df.latitude),
        crs='EPSG:4326'
    )
    
    grid_gdf = gpd.GeoDataFrame(
        features_df,
        geometry=gpd.points_from_xy(features_df.center_lon, features_df.center_lat),
        crs='EPSG:4326'
    )
    
    # Spatial join - find nearest grid cell for each store
    joined = gpd.sjoin_nearest(
        stores_gdf,
        grid_gdf[['cell_id', 'geometry']],
        how='left',
        distance_col='distance_to_cell_center'
    )
    
    print(f"   ✓ Joined {len(joined)} stores to grid cells")
    
    # Merge with full features
    training_data = joined.merge(
        features_df,
        on='cell_id',
        how='left',
        suffixes=('_store', '_grid')
    )
    
    print(f"   ✓ Final training data: {len(training_data)} rows × {len(training_data.columns)} columns")
    
    return training_data


def engineer_features(df):
    """
    Apply feature engineering transformations.
    
    Args:
        df: DataFrame with base features
    
    Returns:
        DataFrame with engineered features added
    """
    print("\n🔧 Engineering features...")
    
    # Add distance to city center
    df = add_city_center_distance(df)
    print("   ✓ Added distance to city center")
    
    # Add interaction features
    df = add_interaction_features(df)
    print("   ✓ Added interaction features")
    
    # Handle missing values
    df = handle_missing_values(df, strategy='median')
    print("   ✓ Handled missing values")
    
    return df


def save_processed_data(df, output_path):
    """Save processed training data to CSV."""
    print(f"\n💾 Saving processed data to {output_path}...")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Drop geometry columns for CSV export
    cols_to_drop = ['geometry', 'geometry_store', 'geometry_grid', 'geometry_wkt']
    df_to_save = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    df_to_save.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df_to_save)} rows × {len(df_to_save.columns)} columns")


def main():
    print("=" * 70)
    print("PREPARING TRAINING DATA FOR ML MODELS")
    print("=" * 70)
    
    # Load database config
    cfg = load_env()
    engine = get_engine(cfg)
    
    # Load features from PostGIS
    features_df = load_features_from_postgis(engine)
    if features_df is None:
        return
    
    # Load training labels
    labels_path = project_root / "data" / "raw" / "real_training_labels.csv"
    labels_df = load_training_labels(labels_path)
    if labels_df is None:
        return
    
    # Spatial join
    training_data = spatial_join_labels_to_grid(labels_df, features_df)
    
    # Feature engineering
    training_data = engineer_features(training_data)
    
    # Save processed data
    output_path = project_root / "data" / "processed" / "training_data.csv"
    save_processed_data(training_data, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total training samples: {len(training_data)}")
    print(f"Total features: {len(training_data.columns)}")
    print(f"\nTarget variable distributions:")
    print(f"  • Success rate: {training_data['success_label'].mean():.1%}")
    print(f"  • Avg monthly revenue: ₹{training_data['monthly_revenue'].mean():,.0f}")
    print(f"  • Avg location score: {training_data['location_score'].mean():.1f}")
    
    print("\n✅ Data preparation complete!")
    print(f"📁 Output: {output_path}")
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Run: python models/train_success_model.py")
    print("2. Run: python models/train_revenue_model.py")
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
