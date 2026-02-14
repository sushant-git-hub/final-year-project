"""
Model Configuration
-------------------
Centralized configuration for ML models, hyperparameters, and feature definitions.
"""

import numpy as np

# ============================================================================
# Feature Definitions
# ============================================================================

NUMERIC_FEATURES = [
    'competitor_count', 'nearest_m',
    'road_density_km', 'major_dist_m',
    'commercial_rent_per_sqft', 'locality_dist_m',
    'total_population', 'ward_dist_m',
    'footfall_generator_count', 'nearest_generator_m',
    'transit_stop_count', 'nearest_transit_m',
    'avg_monthly_income', 'property_price_sqft', 'purchasing_power_index',
    'center_lat', 'center_lon'
]

CATEGORICAL_FEATURES = [
    'tier', 'confidence', 'income_tier'
]

ENGINEERED_FEATURES = [
    'distance_to_city_center',
    'rent_to_income_ratio',
    'transit_accessibility_score',
    'footfall_accessibility_score',
    'competition_density',
    'market_saturation',
    'connectivity_score'
]

# Columns to exclude from training
EXCLUDE_COLUMNS = [
    'cell_id', 'geometry', 'geometry_wkt', 'geometry_store', 'geometry_grid',
    'place_id', 'name', 'latitude', 'longitude',
    'rating', 'user_ratings_total',  # These are used to create labels
    'success_label', 'location_score', 'monthly_revenue', 'daily_footfall',
    'footfall_category', 'profitability',  # Target variables
    'distance_to_cell_center', 'index_right'  # Spatial join artifacts
]

# ============================================================================
# XGBoost Hyperparameters
# ============================================================================

# Binary Classification (success_label)
XGBOOST_CLASSIFIER_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'scale_pos_weight': 1,  # Adjust based on class imbalance
    'random_state': 42,
    'tree_method': 'hist',
    'enable_categorical': False,  # We'll one-hot encode manually
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50
}

# Regression (monthly_revenue, location_score)
XGBOOST_REGRESSOR_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50
}

# ============================================================================
# Hyperparameter Search Space (for tuning)
# ============================================================================

PARAM_SEARCH_SPACE = {
    'n_estimators': [300, 500, 700],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0]
}

# ============================================================================
# Training Configuration
# ============================================================================

TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'cv_folds': 5,
    'stratify': True,  # For classification
    'group_by': 'ward_name',  # For spatial cross-validation
    'scale_features': False,  # XGBoost doesn't require scaling
    'handle_imbalance': True,  # Adjust class weights if needed
}

# ============================================================================
# Model Paths
# ============================================================================

MODEL_DIR = 'models/saved'
METRICS_DIR = 'models/metrics'
PLOTS_DIR = 'models/plots'

MODEL_PATHS = {
    'success_classifier': f'{MODEL_DIR}/xgb_success_classifier.pkl',
    'revenue_regressor': f'{MODEL_DIR}/xgb_revenue_regressor.pkl',
    'location_score_regressor': f'{MODEL_DIR}/xgb_location_score_regressor.pkl',
    'scaler': f'{MODEL_DIR}/feature_scaler.pkl',
    'feature_names': f'{MODEL_DIR}/feature_names.pkl'
}

# ============================================================================
# Evaluation Metrics
# ============================================================================

CLASSIFICATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
]

REGRESSION_METRICS = [
    'r2', 'rmse', 'mae', 'mape'
]

# ============================================================================
# Helper Functions
# ============================================================================

def get_all_features():
    """Get list of all feature columns."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES


def get_feature_columns(df):
    """
    Get feature columns from DataFrame, excluding target and metadata columns.
    
    Args:
        df: DataFrame with all columns
    
    Returns:
        List of feature column names
    """
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in EXCLUDE_COLUMNS]
    return feature_cols


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target variable (binary)
    
    Returns:
        scale_pos_weight parameter for XGBoost
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    
    if n_positive == 0:
        return 1.0
    
    return n_negative / n_positive
