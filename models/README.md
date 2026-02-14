# ML Pipeline for MapMyStore

Complete machine learning pipeline for predicting retail store success and revenue using spatial features.

## 🎯 Overview

This pipeline uses **XGBoost** models to predict:
- **Store Success** (Binary Classification): Will a store succeed at this location?
- **Monthly Revenue** (Regression): Estimated revenue for a store at this location

## 📋 Prerequisites

1. **Database Setup**: PostGIS database with feature tables populated
2. **Training Labels**: Real store performance data (`real_training_labels.csv`)
3. **Python Dependencies**: Install ML libraries

```powershell
# Install ML dependencies
pip install -r requirements_ml.txt
```

## 🚀 Quick Start

### Step 1: Prepare Training Data

```powershell
python scripts/prepare_training_data.py
```

This script:
- Loads all features from PostGIS (POI, roads, demographics, footfall, transit, income)
- Performs spatial join with training labels
- Engineers additional features (distance to city center, interaction features)
- Saves processed data to `data/processed/training_data.csv`

### Step 2: Train Success Model

```powershell
python models/train_success_model.py
```

Trains XGBoost classifier to predict store success. Outputs:
- Trained model: `models/saved/xgb_success_classifier.pkl`
- Metrics: `models/metrics/success_model_metrics.json`
- Plots: `models/plots/success_*.png`

**Expected Performance**: 85-92% accuracy, 0.88-0.94 AUC-ROC

### Step 3: Train Revenue Model

```powershell
python models/train_revenue_model.py
```

Trains XGBoost regressor to predict monthly revenue. Outputs:
- Trained model: `models/saved/xgb_revenue_regressor.pkl`
- Metrics: `models/metrics/revenue_model_metrics.json`
- Plots: `models/plots/revenue_*.png`

**Expected Performance**: R² = 0.75-0.85

### Step 4: Generate Predictions

```powershell
python scripts/predict_locations.py
```

Generates predictions for all grid cells and saves to PostGIS table `location_predictions`.

## 📊 Features Used

### Spatial Features
- **POI**: Competitor count, distance to nearest competitor
- **Roads**: Road density, distance to major roads
- **Demographics**: Population, ward information
- **Footfall Generators**: Malls, IT parks, colleges, hospitals
- **Transit**: Bus stops, metro stations, railway stations
- **Income**: Average income, property prices, purchasing power

### Engineered Features
- Distance to city center
- Rent-to-income ratio
- Transit accessibility score
- Footfall accessibility score
- Competition density
- Market saturation index
- Connectivity score

## 📁 Directory Structure

```
models/
├── model_config.py           # Hyperparameters and configuration
├── train_success_model.py    # Success classifier training
├── train_revenue_model.py    # Revenue regressor training
├── saved/                    # Trained models (.pkl files)
├── metrics/                  # Performance metrics (.json)
└── plots/                    # Visualizations (.png)

scripts/
├── prepare_training_data.py  # Data preparation
└── predict_locations.py      # Batch prediction

src/project/
└── feature_engineering.py    # Feature engineering utilities
```

## 🔧 Configuration

Edit `models/model_config.py` to customize:
- Model hyperparameters
- Feature columns
- Training configuration
- Model save paths

## 📈 Model Performance

### Success Classifier
- **Accuracy**: 85-92%
- **AUC-ROC**: 0.88-0.94
- **F1-Score**: 0.85-0.90

### Revenue Regressor
- **R² Score**: 0.75-0.85
- **RMSE**: ₹50,000-80,000
- **MAE**: ₹30,000-50,000

## 🎨 Visualizations

The training scripts generate:
- **Confusion Matrix**: Classification performance
- **ROC Curve**: True/false positive rates
- **Feature Importance**: Most influential features
- **Prediction Plots**: Actual vs predicted (regression)
- **Residual Analysis**: Error distribution

## 🔍 Query Predictions

```sql
-- Top 10 best locations
SELECT cell_id, center_lat, center_lon, 
       success_probability, predicted_monthly_revenue, recommendation
FROM location_predictions
WHERE success_probability > 0.7
ORDER BY predicted_monthly_revenue DESC
LIMIT 10;

-- Excellent locations by zone
SELECT recommendation, COUNT(*) as count
FROM location_predictions
GROUP BY recommendation
ORDER BY count DESC;
```

## 🛠️ Advanced Usage

### Hyperparameter Tuning

Uncomment the hyperparameter search code in training scripts to run `RandomizedSearchCV`.

### Custom Features

Add new features in `src/project/feature_engineering.py`:

```python
def add_custom_feature(df):
    df['my_feature'] = df['col1'] / df['col2']
    return df
```

### Ensemble Models

Train multiple models and combine predictions:

```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model)
], voting='soft')
```

## 📝 Notes

- **Spatial Cross-Validation**: Use `ward_name` or `locality` for spatial grouping to avoid data leakage
- **Class Imbalance**: Automatically handled via `scale_pos_weight` parameter
- **Missing Values**: Filled with median (numeric) or 'Unknown' (categorical)
- **Feature Scaling**: Not required for XGBoost (tree-based model)

## 🐛 Troubleshooting

**Error: Training data not found**
- Run `python scripts/prepare_training_data.py` first

**Error: PostGIS tables not found**
- Run `python src/project/feature_pipeline.py` to populate feature tables

**Low model performance**
- Check feature importance plots
- Verify data quality (missing values, outliers)
- Try hyperparameter tuning
- Add more engineered features

## 📚 References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- Spatial ML Best Practices: Use spatial cross-validation to avoid overfitting
