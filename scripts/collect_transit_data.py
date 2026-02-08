"""
Automated Transit Accessibility Data Collection
Collects bus stops, metro stations, railway stations from OpenStreetMap
"""

import requests
import pandas as pd
import time
import os

# Configuration
PUNE_BOUNDS = {
    "min_lat": 18.4,
    "max_lat": 18.65,
    "min_lon": 73.7,
    "max_lon": 73.95,
}


def query_osm_overpass(query):
    """Query OpenStreetMap Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': query})
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error querying Overpass API: {response.status_code}")
        return None


def get_bus_stops():
    """Get bus stops from OSM"""
    query = f"""
    [out:json];
    (
      node["highway"="bus_stop"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      node["public_transport"="stop_position"]["bus"="yes"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
    );
    out;
    """
    
    data = query_osm_overpass(query)
    if not data:
        return []
    
    bus_stops = []
    for element in data.get('elements', []):
        if element['type'] != 'node':
            continue
        
        bus_stops.append({
            'name': element.get('tags', {}).get('name', 'Bus Stop'),
            'type': 'Bus Stop',
            'latitude': element['lat'],
            'longitude': element['lon'],
            'operator': element.get('tags', {}).get('operator', 'PMPML'),
            'source': 'OSM'
        })
    
    return bus_stops


def get_metro_stations():
    """Get metro stations from OSM"""
    query = f"""
    [out:json];
    (
      node["railway"="station"]["station"="subway"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      node["railway"="station"]["subway"="yes"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
      node["public_transport"="station"]["subway"="yes"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
    );
    out;
    """
    
    data = query_osm_overpass(query)
    if not data:
        return []
    
    metro_stations = []
    for element in data.get('elements', []):
        if element['type'] != 'node':
            continue
        
        metro_stations.append({
            'name': element.get('tags', {}).get('name', 'Metro Station'),
            'type': 'Metro Station',
            'latitude': element['lat'],
            'longitude': element['lon'],
            'operator': element.get('tags', {}).get('operator', 'Pune Metro'),
            'source': 'OSM'
        })
    
    return metro_stations


def get_railway_stations():
    """Get railway stations from OSM"""
    query = f"""
    [out:json];
    (
      node["railway"="station"]({PUNE_BOUNDS['min_lat']},{PUNE_BOUNDS['min_lon']},{PUNE_BOUNDS['max_lat']},{PUNE_BOUNDS['max_lon']});
    );
    out;
    """
    
    data = query_osm_overpass(query)
    if not data:
        return []
    
    railway_stations = []
    for element in data.get('elements', []):
        if element['type'] != 'node':
            continue
        
        # Skip if it's a metro station (already collected)
        tags = element.get('tags', {})
        if tags.get('station') == 'subway' or tags.get('subway') == 'yes':
            continue
        
        railway_stations.append({
            'name': element.get('tags', {}).get('name', 'Railway Station'),
            'type': 'Railway Station',
            'latitude': element['lat'],
            'longitude': element['lon'],
            'operator': element.get('tags', {}).get('operator', 'Indian Railways'),
            'source': 'OSM'
        })
    
    return railway_stations


def add_known_transit_hubs():
    """Add major transit hubs that might be missing from OSM"""
    known_hubs = [
        # Railway Stations
        {'name': 'Pune Junction', 'type': 'Railway Station', 'latitude': 18.5284, 'longitude': 73.8742, 'operator': 'Indian Railways', 'source': 'Manual'},
        {'name': 'Shivajinagar Railway Station', 'type': 'Railway Station', 'latitude': 18.5314, 'longitude': 73.8446, 'operator': 'Indian Railways', 'source': 'Manual'},
        
        # Metro Stations (Pune Metro operational)
        {'name': 'Civil Court Metro Station', 'type': 'Metro Station', 'latitude': 18.5314, 'longitude': 73.8567, 'operator': 'Pune Metro', 'source': 'Manual'},
        {'name': 'Vanaz Metro Station', 'type': 'Metro Station', 'latitude': 18.5089, 'longitude': 73.8046, 'operator': 'Pune Metro', 'source': 'Manual'},
        {'name': 'Garware College Metro Station', 'type': 'Metro Station', 'latitude': 18.5204, 'longitude': 73.8367, 'operator': 'Pune Metro', 'source': 'Manual'},
        
        # Major Bus Terminals
        {'name': 'Swargate Bus Stand', 'type': 'Bus Terminal', 'latitude': 18.5018, 'longitude': 73.8636, 'operator': 'PMPML', 'source': 'Manual'},
        {'name': 'Wakad Bus Stand', 'type': 'Bus Terminal', 'latitude': 18.5978, 'longitude': 73.7642, 'operator': 'PMPML', 'source': 'Manual'},
        {'name': 'Hadapsar Bus Stand', 'type': 'Bus Terminal', 'latitude': 18.5089, 'longitude': 73.9291, 'operator': 'PMPML', 'source': 'Manual'},
    ]
    
    return known_hubs


def main():
    print("=" * 70)
    print("Collecting Transit Accessibility Data for Pune")
    print("=" * 70)
    
    all_transit = []
    
    # Collect bus stops
    print("\n1. Collecting bus stops from OpenStreetMap...")
    bus_stops = get_bus_stops()
    print(f"   Found {len(bus_stops)} bus stops")
    all_transit.extend(bus_stops)
    time.sleep(2)  # Be nice to OSM servers
    
    # Collect metro stations
    print("\n2. Collecting metro stations from OpenStreetMap...")
    metro_stations = get_metro_stations()
    print(f"   Found {len(metro_stations)} metro stations")
    all_transit.extend(metro_stations)
    time.sleep(2)
    
    # Collect railway stations
    print("\n3. Collecting railway stations from OpenStreetMap...")
    railway_stations = get_railway_stations()
    print(f"   Found {len(railway_stations)} railway stations")
    all_transit.extend(railway_stations)
    time.sleep(2)
    
    # Add known transit hubs
    print("\n4. Adding known transit hubs...")
    known_hubs = add_known_transit_hubs()
    print(f"   Added {len(known_hubs)} known transit hubs")
    all_transit.extend(known_hubs)
    
    # Create DataFrame
    df = pd.DataFrame(all_transit)
    
    if df.empty:
        print("\n⚠️  No data collected. Check your internet connection.")
        return
    
    # Remove duplicates based on name and location
    df['lat_round'] = df['latitude'].round(4)
    df['lon_round'] = df['longitude'].round(4)
    df = df.drop_duplicates(subset=['name', 'lat_round', 'lon_round'])
    df = df.drop(columns=['lat_round', 'lon_round'])
    
    # Save to CSV
    output_path = os.path.join('data', 'raw', 'pune_transit_stops.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total transit stops collected: {len(df)}")
    print(f"\nBreakdown by type:")
    print(df['type'].value_counts().to_string())
    print(f"\nData saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
