"""
ML Model Prediction API - Backend Integration Guide

This script demonstrates how to use the trained ML model to make predictions
for new locations and return results in a format suitable for backend integration.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

class StorePredictionAPI:
    """
    API wrapper for store success prediction model
    """
    
    def __init__(self, model_dir='models/category_specific/optimized'):
        """Initialize the prediction API with trained models"""
        self.model_dir = Path(model_dir)
        
        # Load models
        self.retail_model = joblib.load(self.model_dir / 'retail_optimized_model.pkl')
        self.food_model = joblib.load(self.model_dir / 'food_optimized_model.pkl')
        
        # Load feature names
        self.retail_features = joblib.load(self.model_dir / 'retail_features.pkl')
        self.food_features = joblib.load(self.model_dir / 'food_features.pkl')
        
        print("✓ Models loaded successfully")
    
    def predict_single_location(self, location_data, category='retail'):
        """
        Predict success probability for a single location
        
        Args:
            location_data (dict): Dictionary containing location features
            category (str): 'retail' or 'food'
        
        Returns:
            dict: Prediction results with probability, class, and confidence
        """
        # Select appropriate model and features
        if category == 'retail':
            model = self.retail_model
            required_features = self.retail_features
        else:
            model = self.food_model
            required_features = self.food_features
        
        # Convert to DataFrame
        df = pd.DataFrame([location_data])
        
        # Ensure all required features are present
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select only required features in correct order
        X = df[required_features]
        
        # Make prediction
        prediction_proba = model.predict_proba(X)[0]
        prediction_class = model.predict(X)[0]
        
        # Calculate confidence (distance from 0.5 threshold)
        confidence = abs(prediction_proba[1] - 0.5) * 2  # Scale to 0-1
        
        # Prepare result
        result = {
            'success_probability': float(prediction_proba[1]),  # Probability of success
            'failure_probability': float(prediction_proba[0]),  # Probability of failure
            'predicted_class': int(prediction_class),  # 0 = failure, 1 = success
            'predicted_label': 'SUCCESS' if prediction_class == 1 else 'FAILURE',
            'confidence': float(confidence),  # How confident the model is (0-1)
            'confidence_level': self._get_confidence_level(confidence),
            'category': category
        }
        
        return result
    
    def predict_multiple_locations(self, locations_data, category='retail'):
        """
        Predict success probability for multiple locations
        
        Args:
            locations_data (list): List of dictionaries containing location features
            category (str): 'retail' or 'food'
        
        Returns:
            list: List of prediction results
        """
        results = []
        for location in locations_data:
            result = self.predict_single_location(location, category)
            results.append(result)
        
        return results
    
    def predict_grid_cell(self, latitude, longitude, features, category='retail'):
        """
        Predict success for a specific grid cell (lat/lon)
        
        Args:
            latitude (float): Latitude of location
            longitude (float): Longitude of location
            features (dict): Pre-computed features for this location
            category (str): 'retail' or 'food'
        
        Returns:
            dict: Prediction with location info
        """
        # Add location to features
        location_data = {
            'latitude': latitude,
            'longitude': longitude,
            **features
        }
        
        # Get prediction
        result = self.predict_single_location(location_data, category)
        
        # Add location info
        result['location'] = {
            'latitude': latitude,
            'longitude': longitude
        }
        
        return result
    
    def _get_confidence_level(self, confidence):
        """Convert confidence score to human-readable level"""
        if confidence >= 0.8:
            return 'VERY_HIGH'
        elif confidence >= 0.6:
            return 'HIGH'
        elif confidence >= 0.4:
            return 'MEDIUM'
        elif confidence >= 0.2:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def get_feature_importance(self, category='retail', top_n=10):
        """
        Get top N most important features for the model
        
        Args:
            category (str): 'retail' or 'food'
            top_n (int): Number of top features to return
        
        Returns:
            list: List of (feature_name, importance) tuples
        """
        if category == 'retail':
            model = self.retail_model
            features = self.retail_features
        else:
            model = self.food_model
            features = self.food_features
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create list of (feature, importance) tuples
        feature_importance = list(zip(features, importance))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]


def example_usage():
    """Example of how to use the API"""
    
    print("="*70)
    print("ML MODEL OUTPUT FOR BACKEND INTEGRATION")
    print("="*70)
    
    # Initialize API
    api = StorePredictionAPI()
    
    # Example 1: Single location prediction
    print("\n--- Example 1: Single Location Prediction ---")
    
    location = {
        'latitude': 18.5204,
        'longitude': 73.8567,
        'user_ratings_total': 0,  # New location, no ratings yet
        'distance_to_cell_center': 50.0,
        'center_lat': 18.5200,
        'center_lon': 73.8560,
        'competitor_count': 5,
        'nearest_m': 150.0,
        'road_density_km': 2.5,
        'major_dist_m': 500.0,
        'commercial_rent_per_sqft': 45.0,
        'total_population': 15000,
        'footfall_generator_count': 3,
        'nearest_generator_m': 200.0,
        'transit_stop_count': 4,
        'nearest_transit_m': 100.0,
        'avg_monthly_income': 35000.0,
        'property_price_sqft': 5500.0,
        'purchasing_power_index': 1.2,
        'distance_to_city_center': 5000.0,
        'rent_to_income_ratio': 0.15,
        'transit_accessibility_score': 0.75,
        'footfall_accessibility_score': 0.68,
        'competition_density': 0.025,
        'market_saturation': 0.3,
        'connectivity_score': 0.8,
        # Category-specific features (retail)
        'retail_competition_score': 0.073,
        'rent_affordability': 0.0013,
        'parking_accessibility': 0.75,
        'retail_density': 0.025,
        'visibility_score': 1.7,
        'market_opportunity': 27.2
    }
    
    result = api.predict_single_location(location, category='retail')
    
    print("\nInput Location:")
    print(f"  Latitude: {location['latitude']}")
    print(f"  Longitude: {location['longitude']}")
    print(f"  Competitor Count: {location['competitor_count']}")
    print(f"  Rent: ₹{location['commercial_rent_per_sqft']}/sqft")
    
    print("\nPrediction Output (JSON format for backend):")
    print(json.dumps(result, indent=2))
    
    # Example 2: Multiple locations
    print("\n\n--- Example 2: Multiple Locations (Grid Prediction) ---")
    
    locations = [
        {'latitude': 18.52, 'longitude': 73.85, 'competitor_count': 3, 'commercial_rent_per_sqft': 40},
        {'latitude': 18.53, 'longitude': 73.86, 'competitor_count': 8, 'commercial_rent_per_sqft': 55},
        {'latitude': 18.51, 'longitude': 73.84, 'competitor_count': 2, 'commercial_rent_per_sqft': 35},
    ]
    
    # Add default features (in real use, these would come from your feature pipeline)
    for loc in locations:
        loc.update({
            'user_ratings_total': 0,
            'distance_to_cell_center': 50.0,
            'center_lat': loc['latitude'],
            'center_lon': loc['longitude'],
            'nearest_m': 150.0,
            'road_density_km': 2.5,
            'major_dist_m': 500.0,
            'total_population': 15000,
            'footfall_generator_count': 3,
            'nearest_generator_m': 200.0,
            'transit_stop_count': 4,
            'nearest_transit_m': 100.0,
            'avg_monthly_income': 35000.0,
            'property_price_sqft': 5500.0,
            'purchasing_power_index': 1.2,
            'distance_to_city_center': 5000.0,
            'rent_to_income_ratio': 0.15,
            'transit_accessibility_score': 0.75,
            'footfall_accessibility_score': 0.68,
            'competition_density': 0.025,
            'market_saturation': 0.3,
            'connectivity_score': 0.8,
            'retail_competition_score': 0.073,
            'rent_affordability': 0.0013,
            'parking_accessibility': 0.75,
            'retail_density': 0.025,
            'visibility_score': 1.7,
            'market_opportunity': 27.2
        })
    
    results = api.predict_multiple_locations(locations, category='retail')
    
    print("\nBatch Prediction Results:")
    for i, result in enumerate(results):
        print(f"\nLocation {i+1}:")
        print(f"  Success Probability: {result['success_probability']:.2%}")
        print(f"  Prediction: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence_level']}")
    
    # Example 3: Feature importance
    print("\n\n--- Example 3: Feature Importance ---")
    
    importance = api.get_feature_importance(category='retail', top_n=10)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, imp) in enumerate(importance, 1):
        print(f"  {i}. {feature:40s}: {imp:.4f}")
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    example_usage()
