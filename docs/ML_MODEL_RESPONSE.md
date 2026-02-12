# ML Model Response to Backend - Technical Specification

## 📦 Exact JSON Response

When your backend calls the ML model, it receives this **exact JSON structure**:

```json
{
  "success_probability": 0.8183992505073547,
  "failure_probability": 0.18160074949264526,
  "predicted_class": 1,
  "predicted_label": "SUCCESS",
  "confidence": 0.6367985010147095,
  "confidence_level": "HIGH",
  "category": "retail"
}
```

---

## 🔍 Field-by-Field Breakdown

### 1. `success_probability` (float)
- **Type**: Float (0.0 to 1.0)
- **Example**: `0.8184`
- **Meaning**: Probability that the store will succeed
- **Backend Use**: Multiply by 100 for percentage → `81.84%`

```javascript
// Backend conversion
const successPercent = response.success_probability * 100;
// 0.8184 → 81.84%
```

### 2. `failure_probability` (float)
- **Type**: Float (0.0 to 1.0)
- **Example**: `0.1816`
- **Meaning**: Probability that the store will fail
- **Backend Use**: Usually not shown to user (redundant)
- **Note**: `success_probability + failure_probability = 1.0`

### 3. `predicted_class` (integer)
- **Type**: Integer (0 or 1)
- **Values**: 
  - `0` = Failure (not recommended)
  - `1` = Success (recommended)
- **Backend Use**: Binary decision for filtering/sorting

```javascript
// Backend logic
if (response.predicted_class === 1) {
  showRecommendedBadge();
} else {
  showNotRecommendedBadge();
}
```

### 4. `predicted_label` (string)
- **Type**: String
- **Values**: `"SUCCESS"` or `"FAILURE"`
- **Backend Use**: Display directly to user
- **Note**: Human-readable version of `predicted_class`

### 5. `confidence` (float)
- **Type**: Float (0.0 to 1.0)
- **Example**: `0.6368`
- **Meaning**: How confident the model is (distance from 50/50)
- **Calculation**: `abs(success_probability - 0.5) * 2`
- **Backend Use**: Show confidence indicator to user

```javascript
// Confidence interpretation
if (confidence > 0.8) return "Very confident";
if (confidence > 0.6) return "Confident";
if (confidence > 0.4) return "Moderately confident";
return "Uncertain";
```

### 6. `confidence_level` (string)
- **Type**: String (enum)
- **Values**: 
  - `"VERY_HIGH"` (confidence > 0.8)
  - `"HIGH"` (0.6 - 0.8)
  - `"MEDIUM"` (0.4 - 0.6)
  - `"LOW"` (0.2 - 0.4)
  - `"VERY_LOW"` (< 0.2)
- **Backend Use**: Display confidence badge

### 7. `category` (string)
- **Type**: String
- **Values**: `"retail"` or `"food"`
- **Backend Use**: Track which model was used

---

## 🔄 Complete Request/Response Flow

### **Backend Request to ML Model**

```python
# Python backend example
from models.prediction_api import StorePredictionAPI

predictor = StorePredictionAPI()

# Input data (35 features)
location_data = {
    'latitude': 18.5204,
    'longitude': 73.8567,
    'competitor_count': 5,
    'commercial_rent_per_sqft': 45.0,
    'transit_accessibility_score': 0.75,
    # ... (32 more features)
}

# Call ML model
response = predictor.predict_single_location(
    location_data=location_data,
    category='retail'
)

# response is a Python dict (converts to JSON)
```

### **ML Model Response (Python Dict)**

```python
{
    'success_probability': 0.8183992505073547,
    'failure_probability': 0.18160074949264526,
    'predicted_class': 1,
    'predicted_label': 'SUCCESS',
    'confidence': 0.6367985010147095,
    'confidence_level': 'HIGH',
    'category': 'retail'
}
```

### **Backend Sends to Frontend (JSON)**

```json
{
  "success_probability": 0.8184,
  "failure_probability": 0.1816,
  "predicted_class": 1,
  "predicted_label": "SUCCESS",
  "confidence": 0.6368,
  "confidence_level": "HIGH",
  "category": "retail",
  "location": {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "address": "Koregaon Park, Pune"
  },
  "display": {
    "success_percentage": "81.84%",
    "star_rating": 4,
    "recommendation": "RECOMMENDED",
    "risk_level": "LOW"
  }
}
```

---

## 💻 Backend Processing Examples

### **Example 1: Node.js/Express**

```javascript
// Backend API endpoint
app.post('/api/predict', async (req, res) => {
  const { latitude, longitude, category } = req.body;
  
  // 1. Compute features (from your feature pipeline)
  const features = await computeFeatures(latitude, longitude);
  
  // 2. Call ML model (Python microservice)
  const mlResponse = await fetch('http://ml-service:5000/predict', {
    method: 'POST',
    body: JSON.stringify({
      location_data: features,
      category: category
    })
  });
  
  const prediction = await mlResponse.json();
  
  // 3. ML model returns:
  // {
  //   "success_probability": 0.8184,
  //   "failure_probability": 0.1816,
  //   "predicted_class": 1,
  //   "predicted_label": "SUCCESS",
  //   "confidence": 0.6368,
  //   "confidence_level": "HIGH",
  //   "category": "retail"
  // }
  
  // 4. Enhance response for frontend
  const response = {
    ...prediction,
    location: { latitude, longitude },
    display: {
      success_percentage: `${(prediction.success_probability * 100).toFixed(1)}%`,
      star_rating: Math.round(prediction.success_probability * 5),
      recommendation: prediction.predicted_class === 1 ? 'RECOMMENDED' : 'NOT_RECOMMENDED',
      risk_level: prediction.success_probability > 0.7 ? 'LOW' : 'HIGH'
    }
  };
  
  // 5. Send to frontend
  res.json(response);
});
```

### **Example 2: Python/FastAPI**

```python
from fastapi import FastAPI
from models.prediction_api import StorePredictionAPI

app = FastAPI()
predictor = StorePredictionAPI()

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    # 1. Get features
    features = compute_features(request.latitude, request.longitude)
    
    # 2. Call ML model
    ml_response = predictor.predict_single_location(
        location_data=features,
        category=request.category
    )
    
    # ml_response = {
    #     'success_probability': 0.8184,
    #     'failure_probability': 0.1816,
    #     'predicted_class': 1,
    #     'predicted_label': 'SUCCESS',
    #     'confidence': 0.6368,
    #     'confidence_level': 'HIGH',
    #     'category': 'retail'
    # }
    
    # 3. Enhance for frontend
    response = {
        **ml_response,
        'location': {
            'latitude': request.latitude,
            'longitude': request.longitude
        },
        'display': {
            'success_percentage': f"{ml_response['success_probability'] * 100:.1f}%",
            'star_rating': round(ml_response['success_probability'] * 5),
            'recommendation': 'RECOMMENDED' if ml_response['predicted_class'] == 1 else 'NOT_RECOMMENDED'
        }
    }
    
    return response
```

---

## 📊 Response Data Types

```typescript
// TypeScript interface for ML model response
interface MLModelResponse {
  success_probability: number;      // 0.0 - 1.0
  failure_probability: number;      // 0.0 - 1.0
  predicted_class: 0 | 1;           // 0 = FAILURE, 1 = SUCCESS
  predicted_label: "SUCCESS" | "FAILURE";
  confidence: number;                // 0.0 - 1.0
  confidence_level: "VERY_LOW" | "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH";
  category: "retail" | "food";
}
```

---

## 🎯 What Backend Does With Response

### **1. Store in Database**
```sql
INSERT INTO predictions (
  location_lat,
  location_lon,
  success_probability,
  predicted_class,
  confidence_level,
  category,
  created_at
) VALUES (
  18.5204,
  73.8567,
  0.8184,
  1,
  'HIGH',
  'retail',
  NOW()
);
```

### **2. Send to Frontend**
```javascript
// Add display-friendly fields
const frontendResponse = {
  ...mlResponse,
  successPercentage: "81.84%",
  starRating: 4,
  recommendation: "RECOMMENDED",
  riskLevel: "LOW"
};

res.json(frontendResponse);
```

### **3. Cache Results**
```javascript
// Cache for 1 hour (predictions don't change frequently)
redis.setex(
  `prediction:${lat}:${lon}:${category}`,
  3600,
  JSON.stringify(mlResponse)
);
```

### **4. Log for Analytics**
```javascript
analytics.track('prediction_made', {
  success_probability: mlResponse.success_probability,
  predicted_class: mlResponse.predicted_label,
  confidence: mlResponse.confidence_level,
  category: mlResponse.category
});
```

---

## ⚡ Performance Characteristics

- **Response Time**: 10-50ms (single prediction)
- **Response Size**: ~200 bytes (JSON)
- **Throughput**: 100+ predictions/second
- **Caching**: Recommended for repeated locations

---

## 🔒 Error Handling

### **Possible Errors**

```json
// Error response format
{
  "error": "MISSING_FEATURES",
  "message": "Required feature 'competitor_count' is missing",
  "code": 400
}
```

### **Backend Error Handling**

```javascript
try {
  const prediction = await mlModel.predict(features);
  res.json(prediction);
} catch (error) {
  if (error.code === 'MISSING_FEATURES') {
    res.status(400).json({
      error: 'Invalid input',
      message: error.message
    });
  } else {
    res.status(500).json({
      error: 'Prediction failed',
      message: 'Internal server error'
    });
  }
}
```

---

## 📝 Summary

**ML Model gives backend exactly 7 fields**:

1. `success_probability` → Convert to percentage for display
2. `failure_probability` → Usually ignore (redundant)
3. `predicted_class` → Use for filtering/sorting
4. `predicted_label` → Display to user
5. `confidence` → Show confidence indicator
6. `confidence_level` → Display confidence badge
7. `category` → Track which model was used

**Backend's job**: 
- Receive these 7 fields
- Enhance with display-friendly formats
- Add location metadata
- Send to frontend

**That's it!** Simple, clean, predictable JSON response. 🚀
