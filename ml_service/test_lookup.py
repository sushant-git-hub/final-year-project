
from feature_lookup import FeatureLookup
import json

def test():
    print("Testing FeatureLookup...")
    lookup = FeatureLookup()
    
    # Use the coordinates from the example in docs
    lat = 18.5204
    lon = 73.8567
    
    try:
        features = lookup.get_features_for_location(lat, lon)
        if features:
            print(f"✓ Found {len(features)} features for ({lat}, {lon})")
            print("Sample feature keys:", list(features.keys())[:5])
        else:
            print(f"⚠ No features found for ({lat}, {lon}). The location might be outside the grid.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test()
