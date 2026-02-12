# ML Model Output for Backend Integration

## Quick Summary

Your ML model outputs a **JSON object** with the following structure:

```json
{
  "success_probability": 0.8184,      // 0-1 scale (81.84% chance of success)
  "failure_probability": 0.1816,      // 0-1 scale (18.16% chance of failure)
  "predicted_class": 1,                // 0 = FAILURE, 1 = SUCCESS
  "predicted_label": "SUCCESS",        // Human-readable: "SUCCESS" or "FAILURE"
  "confidence": 0.6368,                // 0-1 scale (how confident the model is)
  "confidence_level": "HIGH",          // VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
  "category": "retail"                 // "retail" or "food"
}
```

---

## Model Accuracy

- **Overall Accuracy**: **72.78%**
- **Retail Model**: 73.50% accuracy
- **Food Model**: 70.49% accuracy

This means the model correctly predicts success/failure **~73% of the time**.

---

## How to Use in Your Backend

### Option 1: Python Backend (Flask/FastAPI)

```python
from models.prediction_api import StorePredictionAPI

# Initialize once at startup
predictor = StorePredictionAPI()

# In your API endpoint
@app.post("/api/predict-location")
def predict_location(location_data: dict):
    # location_data contains latitude, longitude, and features
    result = predictor.predict_single_location(
        location_data, 
        category='retail'  # or 'food'
    )
    return result
```

### Option 2: REST API Response

```javascript
// Frontend calls your backend
fetch('/api/predict-location', {
  method: 'POST',
  body: JSON.stringify({
    latitude: 18.5204,
    longitude: 73.8567,
    competitor_count: 5,
    commercial_rent_per_sqft: 45,
    // ... other features
  })
})
.then(res => res.json())
.then(data => {
  console.log(data.success_probability);  // 0.8184
  console.log(data.predicted_label);       // "SUCCESS"
  console.log(data.confidence_level);      // "HIGH"
});
```

---

## Required Input Features (35 total)

Your backend needs to provide these features for each location:

### Location Features (4)
- `latitude` - Location latitude
- `longitude` - Location longitude  
- `user_ratings_total` - Number of reviews (0 for new locations)
- `distance_to_cell_center` - Distance to grid cell center (meters)

### Grid Cell Features (2)
- `center_lat` - Grid cell center latitude
- `center_lon` - Grid cell center longitude

### Competition Features (3)
- `competitor_count` - Number of competitors nearby
- `nearest_m` - Distance to nearest competitor (meters)
- `competition_density` - Competitors per square km

### Infrastructure Features (4)
- `road_density_km` - Road density (km/sq km)
- `major_dist_m` - Distance to major road (meters)
- `transit_stop_count` - Number of transit stops nearby
- `nearest_transit_m` - Distance to nearest transit (meters)

### Economic Features (7)
- `commercial_rent_per_sqft` - Rent in ₹/sqft
- `total_population` - Population in area
- `avg_monthly_income` - Average income (₹)
- `property_price_sqft` - Property price (₹/sqft)
- `purchasing_power_index` - Purchasing power (0-2 scale)
- `rent_to_income_ratio` - Rent/income ratio
- `distance_to_city_center` - Distance to city center (meters)

### Footfall Features (3)
- `footfall_generator_count` - Malls, colleges, etc. nearby
- `nearest_generator_m` - Distance to nearest generator (meters)
- `footfall_accessibility_score` - Footfall score (0-1)

### Derived Features (6)
- `transit_accessibility_score` - Transit score (0-1)
- `market_saturation` - Market saturation (0-1)
- `connectivity_score` - Connectivity (0-1)

### Category-Specific Features (6 for retail, 6 for food)

**Retail**:
- `retail_competition_score`
- `rent_affordability`
- `parking_accessibility`
- `retail_density`
- `visibility_score`
- `market_opportunity`

**Food**:
- `residential_proximity`
- `office_accessibility`
- `transit_access`
- `evening_potential`
- `lunch_potential`
- `food_connectivity`

---

## Interpreting the Output

### Success Probability
- **> 0.8 (80%)**: Very likely to succeed ✅
- **0.6 - 0.8 (60-80%)**: Likely to succeed ✅
- **0.4 - 0.6 (40-60%)**: Uncertain ⚠️
- **0.2 - 0.4 (20-40%)**: Likely to fail ❌
- **< 0.2 (20%)**: Very likely to fail ❌

### Confidence Level
- **VERY_HIGH**: Model is very confident (>80%)
- **HIGH**: Model is confident (60-80%)
- **MEDIUM**: Model is moderately confident (40-60%)
- **LOW**: Model is uncertain (20-40%)
- **VERY_LOW**: Model is very uncertain (<20%)

**Important**: Even with HIGH confidence, the model can be wrong ~27% of the time!

---

## Example Backend Integration (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.prediction_api import StorePredictionAPI
from typing import Optional

app = FastAPI()

# Initialize model at startup
predictor = StorePredictionAPI()

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    category: str = 'retail'
    features: dict  # All 35 features

class PredictionResponse(BaseModel):
    success_probability: float
    failure_probability: float
    predicted_class: int
    predicted_label: str
    confidence: float
    confidence_level: str
    category: str
    location: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict_location(request: LocationRequest):
    """
    Predict store success for a given location
    """
    try:
        # Combine location and features
        location_data = {
            'latitude': request.latitude,
            'longitude': request.longitude,
            **request.features
        }
        
        # Get prediction
        result = predictor.predict_single_location(
            location_data, 
            category=request.category
        )
        
        # Add location info
        result['location'] = {
            'latitude': request.latitude,
            'longitude': request.longitude
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(locations: list[LocationRequest]):
    """
    Predict for multiple locations
    """
    results = []
    for loc in locations:
        result = await predict_location(loc)
        results.append(result)
    
    return {"predictions": results}

@app.get("/feature-importance/{category}")
async def get_feature_importance(category: str, top_n: int = 10):
    """
    Get most important features for the model
    """
    importance = predictor.get_feature_importance(category, top_n)
    return {
        "category": category,
        "top_features": [
            {"feature": f, "importance": float(imp)} 
            for f, imp in importance
        ]
    }
```

---

## Testing the API

```bash
# Start your backend
uvicorn main:app --reload

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 18.5204,
    "longitude": 73.8567,
    "category": "retail",
    "features": {
      "competitor_count": 5,
      "commercial_rent_per_sqft": 45,
      ...
    }
  }'
```

---

## Important Notes

### 1. Feature Computation
Your backend needs to compute all 35 features before calling the model. Use the feature pipeline:

```python
from scripts.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline()
features = pipeline.compute_features(latitude, longitude)
```

### 2. Model Files Required
Make sure these files are accessible to your backend:
- `models/category_specific/optimized/retail_optimized_model.pkl`
- `models/category_specific/optimized/food_optimized_model.pkl`
- `models/category_specific/optimized/retail_features.pkl`
- `models/category_specific/optimized/food_features.pkl`

### 3. Dependencies
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### 4. Performance
- Single prediction: ~10-50ms
- Batch of 100: ~500ms-1s
- Consider caching for repeated locations

---

## Frontend Display Example

```javascript
// Display prediction to user
function displayPrediction(result) {
  const successRate = (result.success_probability * 100).toFixed(1);
  const confidence = result.confidence_level;
  
  return `
    <div class="prediction-result">
      <h3>${result.predicted_label}</h3>
      <p>Success Probability: ${successRate}%</p>
      <p>Confidence: ${confidence}</p>
      
      ${result.predicted_label === 'SUCCESS' 
        ? '<span class="badge success">✓ Recommended Location</span>'
        : '<span class="badge warning">⚠ Not Recommended</span>'
      }
    </div>
  `;
}
```

---

## Summary

**What your ML model gives you**:
1. ✅ Success probability (0-1 scale)
2. ✅ Success/Failure prediction
3. ✅ Confidence level
4. ✅ JSON format ready for API

**What you need to provide**:
1. 📍 Location (lat/lon)
2. 📊 35 computed features
3. 🏷️ Category (retail/food)

**Accuracy**: ~73% correct predictions

The model is ready for production use! 🚀
