"""
Feature Engineering Utilities for MapMyStore
---------------------------------------------
Spatial feature engineering functions for ML model training and inference.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import RobustScaler, StandardScaler


def calculate_distance_to_point(lat, lon, target_lat, target_lon):
    """
    Calculate distance in meters between two lat/lon points using Haversine formula.
    
    Args:
        lat, lon: Source coordinates
        target_lat, target_lon: Target coordinates
    
    Returns:
        Distance in meters
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Earth radius in meters
    
    lat1, lon1 = radians(lat), radians(lon)
    lat2, lon2 = radians(target_lat), radians(target_lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def add_city_center_distance(df, city_center_lat=18.5204, city_center_lon=73.8567):
    """
    Add distance to Pune city center (Shivajinagar).
    
    Args:
        df: DataFrame with 'center_lat' and 'center_lon' columns
        city_center_lat, city_center_lon: Pune city center coordinates
    
    Returns:
        DataFrame with 'distance_to_city_center' column
    """
    df = df.copy()
    df['distance_to_city_center'] = df.apply(
        lambda row: calculate_distance_to_point(
            row['center_lat'], row['center_lon'],
            city_center_lat, city_center_lon
        ),
        axis=1
    )
    return df


def add_interaction_features(df):
    """
    Add interaction features between existing columns.
    
    Args:
        df: DataFrame with base features
    
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    # Rent to income ratio
    if 'commercial_rent_per_sqft' in df.columns and 'avg_monthly_income' in df.columns:
        df['rent_to_income_ratio'] = df['commercial_rent_per_sqft'] / (df['avg_monthly_income'] + 1)
    
    # Transit accessibility score (inverse distance weighted by count)
    if 'nearest_transit_m' in df.columns and 'transit_stop_count' in df.columns:
        df['transit_accessibility_score'] = (
            df['transit_stop_count'] / (1 + df['nearest_transit_m'] / 1000)
        )
    
    # Footfall accessibility score
    if 'nearest_generator_m' in df.columns and 'footfall_generator_count' in df.columns:
        df['footfall_accessibility_score'] = (
            df['footfall_generator_count'] / (1 + df['nearest_generator_m'] / 1000)
        )
    
    # Competition density (competitors per population)
    if 'competitor_count' in df.columns and 'total_population' in df.columns:
        df['competition_density'] = df['competitor_count'] / (df['total_population'] / 1000 + 1)
    
    # Market saturation index
    if 'competitor_count' in df.columns and 'total_population' in df.columns:
        df['market_saturation'] = df['competitor_count'] / (df['total_population'] / 10000 + 1)
    
    # Connectivity score (road density × inverse distance to major road)
    if 'road_density_km' in df.columns and 'major_dist_m' in df.columns:
        df['connectivity_score'] = df['road_density_km'] / (1 + df['major_dist_m'] / 1000)
    
    return df


def add_density_features(df, radius_m=1000):
    """
    Add density-based features within a radius.
    Note: This requires spatial operations and is simplified here.
    For production, use PostGIS ST_Buffer and spatial aggregations.
    
    Args:
        df: DataFrame with spatial features
        radius_m: Radius in meters for density calculations
    
    Returns:
        DataFrame with density features
    """
    df = df.copy()
    
    # Placeholder for density features
    # In production, these would be computed via PostGIS queries
    # Example: SELECT COUNT(*) FROM competitors WHERE ST_DWithin(cell.geom, competitor.geom, 1000)
    
    return df


def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in feature DataFrame.
    
    Args:
        df: DataFrame with features
        strategy: 'median', 'mean', or 'zero'
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isna().any():
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'zero':
                df[col].fillna(0, inplace=True)
    
    # Handle categorical missing values
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col].fillna('Unknown', inplace=True)
    
    return df


def scale_features(df, feature_cols, method='robust'):
    """
    Scale numeric features.
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to scale
        method: 'robust' (for outliers) or 'standard'
    
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    df = df.copy()
    
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # Only scale numeric columns that exist
    cols_to_scale = [col for col in feature_cols if col in df.columns and df[col].dtype in [np.float64, np.int64]]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df, scaler


def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df: DataFrame with features
        categorical_cols: List of categorical column names
    
    Returns:
        DataFrame with encoded features
    """
    df = df.copy()
    
    # Filter to only existing columns
    existing_cats = [col for col in categorical_cols if col in df.columns]
    
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    
    return df


def prepare_features_for_training(df, target_col=None, categorical_cols=None, scale=True):
    """
    Complete feature preparation pipeline.
    
    Args:
        df: Raw feature DataFrame
        target_col: Name of target column (will be excluded from features)
        categorical_cols: List of categorical columns to encode
        scale: Whether to scale numeric features
    
    Returns:
        Tuple of (X, y, feature_names, scaler)
    """
    df = df.copy()
    
    # Add engineered features
    df = add_city_center_distance(df)
    df = add_interaction_features(df)
    
    # Handle missing values
    df = handle_missing_values(df, strategy='median')
    
    # Separate features and target
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    
    # Remove non-feature columns
    cols_to_drop = ['cell_id', 'geometry', 'place_id', 'name', 'latitude', 'longitude']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
    
    # Encode categorical features
    if categorical_cols:
        X = encode_categorical_features(X, categorical_cols)
    
    # Scale features
    scaler = None
    if scale:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X, scaler = scale_features(X, numeric_cols, method='robust')
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, scaler


def get_feature_columns():
    """
    Define standard feature columns for the model.
    
    Returns:
        Dictionary with feature column lists
    """
    return {
        'numeric': [
            'competitor_count', 'nearest_m',
            'road_density_km', 'major_dist_m',
            'commercial_rent_per_sqft', 'locality_dist_m',
            'total_population', 'ward_dist_m',
            'footfall_generator_count', 'nearest_generator_m',
            'transit_stop_count', 'nearest_transit_m',
            'avg_monthly_income', 'property_price_sqft', 'purchasing_power_index',
            'center_lat', 'center_lon'
        ],
        'categorical': [
            # Removed location-specific features: 'zone', 'locality', 'ward_id', 'ward_name'
            # These prevent generalization to new areas
            'tier', 'confidence', 'income_tier'
        ],
        'engineered': [
            'distance_to_city_center',
            'rent_to_income_ratio',
            'transit_accessibility_score',
            'footfall_accessibility_score',
            'competition_density',
            'market_saturation',
            'connectivity_score'
        ]
    }
