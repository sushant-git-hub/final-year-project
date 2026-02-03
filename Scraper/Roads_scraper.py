import osmnx as ox
import pandas as pd
import numpy as np

# Configure OSMnx settings
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 300

print("Downloading Pune road network from OpenStreetMap...")
print("This may take several minutes, please be patient...\n")

try:
    # Download road network
    G = ox.graph_from_place(
        "Pune, Maharashtra, India", 
        network_type='drive',
        simplify=True,
        retain_all=False
    )
    
    print("✓ Successfully downloaded road network!")
    
    # Convert to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)
    
    print(f"✓ Processed {len(edges)} road segments\n")
    
    # Create a clean dataframe with basic info
    roads_data = []
    
    print("Processing road data...")
    for idx, row in edges.iterrows():
        road_dict = {}
        
        # Basic identifiers
        road_dict['osmid'] = str(row.get('osmid', ''))
        road_dict['name'] = str(row.get('name', 'Unnamed'))
        
        # Highway type - handle lists/arrays
        highway = row.get('highway', 'unclassified')
        if isinstance(highway, (list, tuple, np.ndarray)):
            highway = highway[0] if len(highway) > 0 else 'unclassified'
        road_dict['highway'] = str(highway)
        
        # Length
        road_dict['length'] = float(row.get('length', 0))
        
        # Optional attributes
        road_dict['lanes'] = str(row.get('lanes', ''))
        road_dict['maxspeed'] = str(row.get('maxspeed', ''))
        road_dict['oneway'] = str(row.get('oneway', ''))
        road_dict['ref'] = str(row.get('ref', ''))
        road_dict['surface'] = str(row.get('surface', ''))
        
        # Extract coordinates from geometry
        if hasattr(row, 'geometry') and row.geometry is not None:
            coords = list(row.geometry.coords)
            if len(coords) > 0:
                road_dict['start_lon'] = coords[0][0]
                road_dict['start_lat'] = coords[0][1]
                road_dict['end_lon'] = coords[-1][0]
                road_dict['end_lat'] = coords[-1][1]
                road_dict['geometry_wkt'] = row.geometry.wkt
            else:
                road_dict['start_lon'] = ''
                road_dict['start_lat'] = ''
                road_dict['end_lon'] = ''
                road_dict['end_lat'] = ''
                road_dict['geometry_wkt'] = ''
        else:
            road_dict['start_lon'] = ''
            road_dict['start_lat'] = ''
            road_dict['end_lon'] = ''
            road_dict['end_lat'] = ''
            road_dict['geometry_wkt'] = ''
        
        roads_data.append(road_dict)
    
    # Create DataFrame
    roads_df = pd.DataFrame(roads_data)
    
    print("✓ Data extraction complete")
    
    # Add road classification ranking
    highway_order = {
        'motorway': 1,
        'trunk': 2,
        'primary': 3,
        'secondary': 4,
        'tertiary': 5,
        'unclassified': 6,
        'residential': 7,
        'motorway_link': 8,
        'trunk_link': 9,
        'primary_link': 10,
        'secondary_link': 11,
        'tertiary_link': 12,
        'living_street': 13
    }
    
    roads_df['road_size_rank'] = roads_df['highway'].map(highway_order).fillna(99).astype(int)
    
    # Add readable category
    def categorize_road(rank):
        if rank <= 2:
            return 'Major Highway'
        elif rank <= 4:
            return 'Primary Road'
        elif rank <= 6:
            return 'Secondary Road'
        elif rank <= 8:
            return 'Tertiary Road'
        else:
            return 'Local/Service Road'
    
    roads_df['road_category'] = roads_df['road_size_rank'].apply(categorize_road)
    
    # Sort by road size (1 = biggest)
    roads_df = roads_df.sort_values('road_size_rank')
    
    # Save to CSV
    output_file = 'pune_roads_data.csv'
    roads_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ Data successfully saved to '{output_file}'\n")
    
    # Summary statistics
    print("=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"Total road segments: {len(roads_df)}")
    
    total_length_km = roads_df['length'].sum() / 1000
    print(f"Total road length: {total_length_km:.2f} km\n")
    
    print("Road Types Distribution:")
    print("-" * 50)
    highway_counts = roads_df['highway'].value_counts()
    for road_type, count in highway_counts.items():
        percentage = (count / len(roads_df)) * 100
        print(f"  {road_type:<20} {count:>6} ({percentage:>5.1f}%)")
    
    print("\nRoad Categories:")
    print("-" * 50)
    category_counts = roads_df['road_category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(roads_df)) * 100
        print(f"  {category:<20} {count:>6} ({percentage:>5.1f}%)")
    
    print(f"\nColumns in CSV file:")
    print(f"  {', '.join(roads_df.columns)}")
    print("\n" + "=" * 50)
    print("\n✓ SUCCESS! Your CSV file is ready to use.")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}\n")
    import traceback
    print("Full error details:")
    traceback.print_exc()
    
    print("\nALTERNATIVE SOLUTIONS:")
    print("-" * 50)
    print("\n1. Download pre-extracted data from Geofabrik:")
    print("   https://download.geofabrik.de/asia/india/maharashtra-latest.osm.pbf")
    print("   Then use osmium or QGIS to filter Pune area\n")
    
    print("2. Use BBBike Extract:")
    print("   https://extract.bbbike.org/")
    print("   Select Pune area manually and download as Shapefile\n")
    
    print("3. Try running the script again - sometimes it's just a timing issue")