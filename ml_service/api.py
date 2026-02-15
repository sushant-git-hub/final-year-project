import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Add project root to path to import models
# This assumes ml_service is at the root of the project
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# ... imports ...
try:
    from models.prediction_api import StorePredictionAPI
except ImportError:
    try:
        from models.prediction_api import StorePredictionAPI
    except ImportError as e:
        print(f"Error importing models: {e}")
        # raise # Don't raise yet, as we might be setting up
        StorePredictionAPI = None

from feature_lookup import FeatureLookup

app = FastAPI(title="Store Prediction API")

# Initialize feature lookup
feature_lookup = FeatureLookup()

# Initialize predictor
try:
    if StorePredictionAPI:
        model_path = project_root / 'models' / 'category_specific' / 'optimized'
        predictor = StorePredictionAPI(model_dir=str(model_path))
    else:
        predictor = None
except Exception as e:
    print(f"Failed to initialize predictor: {e}")
    predictor = None

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    category: str = 'retail'
    features: Optional[Dict[str, Any]] = None

class BatchRequest(BaseModel):
    locations: List[LocationRequest]

@app.get("/")
def read_root():
    return {"status": "online", "service": "Store Prediction API"}

@app.post("/predict")
async def predict_location(request: LocationRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # 1. Look up features from DB
        db_features = feature_lookup.get_features_for_location(request.latitude, request.longitude)
        
        if db_features is None:
             # If location is outside our grid, we can either fail or proceed with defaults/provided features
             # For now, let's assuming if no DB features, we rely entirely on request.features or fail
             if not request.features:
                 raise HTTPException(status_code=404, detail="Location outside of service area (no grid data found)")
             db_features = {}

        # 2. Merge with request features (overrides)
        features = db_features.copy()
        if request.features:
            features.update(request.features)
            
        # 3. Combine location and features
        location_data = {
            'latitude': request.latitude,
            'longitude': request.longitude,
            **features
        }
        
        # 4. Get prediction
        result = predictor.predict_single_location(
            location_data, 
            category=request.category
        )
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(request: BatchRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # 1. Batch lookup features
    db_features_list = feature_lookup.get_features_for_batch(request.locations)
    
    results = []
    for i, loc in enumerate(request.locations):
        try:
            # Get pre-fetched features
            db_features = db_features_list[i]
            if db_features is None:
                db_features = {}
            
            # 2. Merge
            features = db_features.copy()
            if loc.features:
                features.update(loc.features)

            location_data = {
                'latitude': loc.latitude,
                'longitude': loc.longitude,
                **features
            }
            
            # 3. Predict
            result = predictor.predict_single_location(
                location_data, 
                category=loc.category
            )
            
            # Add back location info
            result['location'] = {
                'latitude': loc.latitude,
                'longitude': loc.longitude
            }
            results.append(result)
        except Exception as e:
             results.append({"error": str(e), "latitude": loc.latitude, "longitude": loc.longitude})
    
    return {"predictions": results}

@app.get("/feature-importance/{category}")
async def get_feature_importance(category: str, top_n: int = 10):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        importance = predictor.get_feature_importance(category, top_n)
        return {
            "category": category,
            "top_features": [
                {"feature": f, "importance": float(imp)} 
                for f, imp in importance
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
