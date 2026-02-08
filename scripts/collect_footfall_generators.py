"""
Automated Footfall Generator Data Collection
Collects malls, IT parks, colleges, hospitals using OpenStreetMap and Google Places API
"""

import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
PUNE_BOUNDS = {
    "min_lat": 18.4,
    "max_lat": 18.65,
    "min_lon": 73.7,
    "max_lon": 73.95,
}

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")


def query_osm_overpass(query):
    """Query OpenStreetMap Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': query})
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error querying Overpass API: {response.status_code}")
        return None


def get_malls_osm():
    """Get shopping malls from OSM"""
    query = f"""
    [out:json];
    (
      node["shop"="mall"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      way["shop"="mall"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      relation["shop"="mall"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
    );
    out center;
    """
    
    data = query_osm_overpass(query)
    if not data:
        return []
    
    malls = []
    for element in data.get('elements', []):
        if element['type'] == 'node':
            lat, lon = element['lat'], element['lon']
        elif 'center' in element:
            lat, lon = element['center']['lat'], element['center']['lon']
        else:
            continue
        
        malls.append({
            'name': element.get('tags', {}).get('name', 'Unknown Mall'),
            'type': 'Mall',
            'latitude': lat,
            'longitude': lon,
            'importance': 4,  # Default importance
            'source': 'OSM'
        })
    
    return malls


def get_universities_osm():
    """Get universities and colleges from OSM"""
    query = f"""
    [out:json];
    (
      node["amenity"="university"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      node["amenity"="college"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      way["amenity"="university"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      way["amenity"="college"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
    );
    out center;
    """
    
    data = query_osm_overpass(query)
    if not data:
        return []
    
    colleges = []
    for element in data.get('elements', []):
        if element['type'] == 'node':
            lat, lon = element['lat'], element['lon']
        elif 'center' in element:
            lat, lon = element['center']['lat'], element['center']['lon']
        else:
            continue
        
        colleges.append({
            'name': element.get('tags', {}).get('name', 'Unknown College'),
            'type': 'College',
            'latitude': lat,
            'longitude': lon,
            'importance': 3,
            'source': 'OSM'
        })
    
    return colleges


def get_hospitals_osm():
    """Get hospitals from OSM"""
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      way["amenity"="hospital"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
    );
    out center;
    """
    
    data = query_osm_overpass(query)
    if not data:
        return []
    
    hospitals = []
    for element in data.get('elements', []):
        if element['type'] == 'node':
            lat, lon = element['lat'], element['lon']
        elif 'center' in element:
            lat, lon = element['center']['lat'], element['center']['lon']
        else:
            continue
        
        hospitals.append({
            'name': element.get('tags', {}).get('name', 'Unknown Hospital'),
            'type': 'Hospital',
            'latitude': lat,
            'longitude': lon,
            'importance': 3,
            'source': 'OSM'
        })
    
    return hospitals


def get_it_parks_google():
    """Get IT parks using Google Places API (if available)"""
    if not GOOGLE_API_KEY:
        print("Google API key not found. Skipping IT parks from Google.")
        return []
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    
    it_parks = []
    queries = [
        "IT park Pune",
        "Tech park Pune",
        "Software park Pune",
        "Business park Pune"
    ]
    
    for query in queries:
        params = {
            'query': query,
            'key': GOOGLE_API_KEY
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            continue
        
        data = response.json()
        for place in data.get('results', []):
            location = place['geometry']['location']
            it_parks.append({
                'name': place.get('name', 'Unknown IT Park'),
                'type': 'IT Park',
                'latitude': location['lat'],
                'longitude': location['lng'],
                'importance': 5,
                'source': 'Google'
            })
        
        time.sleep(0.5)  # Rate limiting
    
    return it_parks


def add_manual_known_locations():
    """Add well-known locations that might be missed by automated queries"""
    known_locations = [
        # Major Malls
        {'name': 'Phoenix Marketcity', 'type': 'Mall', 'latitude': 18.5593, 'longitude': 73.7772, 'importance': 5, 'source': 'Manual'},
        {'name': 'Seasons Mall', 'type': 'Mall', 'latitude': 18.5204, 'longitude': 73.8567, 'importance': 4, 'source': 'Manual'},
        {'name': 'Amanora Town Centre', 'type': 'Mall', 'latitude': 18.5089, 'longitude': 73.9208, 'importance': 5, 'source': 'Manual'},
        {'name': 'Westend Mall', 'type': 'Mall', 'latitude': 18.5314, 'longitude': 73.8446, 'importance': 3, 'source': 'Manual'},
        
        # Major IT Parks
        {'name': 'Rajiv Gandhi Infotech Park', 'type': 'IT Park', 'latitude': 18.5913, 'longitude': 73.7371, 'importance': 5, 'source': 'Manual'},
        {'name': 'Magarpatta City', 'type': 'IT Park', 'latitude': 18.5158, 'longitude': 73.9291, 'importance': 5, 'source': 'Manual'},
        {'name': 'EON Free Zone', 'type': 'IT Park', 'latitude': 18.5642, 'longitude': 73.7710, 'importance': 4, 'source': 'Manual'},
        {'name': 'Hinjewadi IT Park Phase 1', 'type': 'IT Park', 'latitude': 18.5913, 'longitude': 73.7371, 'importance': 5, 'source': 'Manual'},
        
        # Major Colleges
        {'name': 'COEP', 'type': 'College', 'latitude': 18.5287, 'longitude': 73.8673, 'importance': 4, 'source': 'Manual'},
        {'name': 'VIT Pune', 'type': 'College', 'latitude': 18.4634, 'longitude': 73.8671, 'importance': 3, 'source': 'Manual'},
        {'name': 'Symbiosis', 'type': 'College', 'latitude': 18.5089, 'longitude': 73.8046, 'importance': 4, 'source': 'Manual'},
        {'name': 'Pune University', 'type': 'College', 'latitude': 18.5466, 'longitude': 73.8252, 'importance': 5, 'source': 'Manual'},
        
        # Major Hospitals
        {'name': 'Ruby Hall Clinic', 'type': 'Hospital', 'latitude': 18.5204, 'longitude': 73.8567, 'importance': 4, 'source': 'Manual'},
        {'name': 'Jehangir Hospital', 'type': 'Hospital', 'latitude': 18.5314, 'longitude': 73.8446, 'importance': 4, 'source': 'Manual'},
        {'name': 'Sahyadri Hospital', 'type': 'Hospital', 'latitude': 18.5089, 'longitude': 73.8046, 'importance': 3, 'source': 'Manual'},
    ]
    
    return known_locations


def main():
    print("=" * 70)
    print("Collecting Footfall Generator Data for Pune")
    print("=" * 70)
    
    all_generators = []
    
    # Collect from OSM
    print("\n1. Collecting malls from OpenStreetMap...")
    malls = get_malls_osm()
    print(f"   Found {len(malls)} malls")
    all_generators.extend(malls)
    time.sleep(1)
    
    print("\n2. Collecting colleges/universities from OpenStreetMap...")
    colleges = get_universities_osm()
    print(f"   Found {len(colleges)} educational institutions")
    all_generators.extend(colleges)
    time.sleep(1)
    
    print("\n3. Collecting hospitals from OpenStreetMap...")
    hospitals = get_hospitals_osm()
    print(f"   Found {len(hospitals)} hospitals")
    all_generators.extend(hospitals)
    time.sleep(1)
    
    # Collect IT parks from Google (if API key available)
    print("\n4. Collecting IT parks from Google Places...")
    it_parks = get_it_parks_google()
    print(f"   Found {len(it_parks)} IT parks")
    all_generators.extend(it_parks)
    
    # Add known locations
    print("\n5. Adding well-known locations...")
    known = add_manual_known_locations()
    print(f"   Added {len(known)} known locations")
    all_generators.extend(known)
    
    # Create DataFrame and remove duplicates
    df = pd.DataFrame(all_generators)
    
    if df.empty:
        print("\n⚠️  No data collected. Check your internet connection.")
        return
    
    # Remove duplicates based on name and location
    df['lat_round'] = df['latitude'].round(4)
    df['lon_round'] = df['longitude'].round(4)
    df = df.drop_duplicates(subset=['name', 'lat_round', 'lon_round'])
    df = df.drop(columns=['lat_round', 'lon_round'])
    
    # Save to CSV
    output_path = os.path.join('data', 'raw', 'pune_footfall_generators.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total footfall generators collected: {len(df)}")
    print(f"\nBreakdown by type:")
    print(df['type'].value_counts().to_string())
    print(f"\nData saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
