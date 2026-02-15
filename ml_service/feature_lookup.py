
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FeatureLookup:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "password")
        self.dbname = os.getenv("DB_NAME", "mapmystore")
        
    def get_connection(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname
        )

    def get_features_for_location(self, latitude, longitude):
        """
        Fetch features for a given latitude/longitude from PostGIS
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Find the grid cell for this location
                # We use ST_Contains to find which grid cell contains the point
                query = """
                SELECT cell_id 
                FROM grid_cells 
                WHERE ST_Contains(
                    geometry, 
                    ST_SetSRID(ST_Point(%s, %s), 4326)
                )
                LIMIT 1;
                """
                cur.execute(query, (longitude, latitude))
                result = cur.fetchone()
                
                if not result:
                    return None  # Location not in our grid
                
                cell_id = result['cell_id']
                
                # 2. Fetch all features for this cell
                features = {}
                
                # Helper to fetch and merge features
                def fetch_table_features(table_name):
                    try:
                        cur.execute(f"SELECT * FROM {table_name} WHERE cell_id = %s", (cell_id,))
                        row = cur.fetchone()
                        if row:
                            # Exclude non-feature columns
                            for key, value in row.items():
                                if key not in ['cell_id', 'geometry', 'id']:
                                    features[key] = value
                    except Exception as e:
                        print(f"Error fetching from {table_name}: {e}")
                
                # Fetch from all feature tables
                fetch_table_features('poi_features')
                fetch_table_features('road_features')
                fetch_table_features('rental_features')
                fetch_table_features('demographic_features')
                fetch_table_features('footfall_features')
                fetch_table_features('transit_features')
                fetch_table_features('income_features')
                
                # 3. Add derived features / clean up
                # Ensure we have all 35 required features with correct types
                # (This list should match what the model expects)
                
                # Fill missing values with 0 or appropriate defaults
                # (Simple fill for now, more complex imputation could be added if needed)
                for key, value in features.items():
                    if value is None:
                        features[key] = 0
                        
                return features
                
        finally:
            conn.close()

    def get_features_for_batch(self, locations):
        """
        Fetch features for a list of locations efficiently
        """
        conn = self.get_connection()
        try:
            results = []
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for loc in locations:
                    lat = loc.latitude
                    lon = loc.longitude
                    
                    # 1. Find the grid cell
                    query = """
                    SELECT cell_id 
                    FROM grid_cells 
                    WHERE ST_Contains(
                        geometry, 
                        ST_SetSRID(ST_Point(%s, %s), 4326)
                    )
                    LIMIT 1;
                    """
                    cur.execute(query, (lon, lat))
                    result = cur.fetchone()
                    
                    if not result:
                        results.append(None)
                        continue
                    
                    cell_id = result['cell_id']
                    features = {}
                    
                    # Helper to fetch features (reusing cursor)
                    def fetch_table_features(table_name):
                        try:
                            cur.execute(f"SELECT * FROM {table_name} WHERE cell_id = %s", (cell_id,))
                            row = cur.fetchone()
                            if row:
                                for key, value in row.items():
                                    if key not in ['cell_id', 'geometry', 'id']:
                                        features[key] = value
                        except Exception as e:
                            print(f"Error fetching from {table_name} for batch: {e}")

                    # Fetch from all feature tables
                    fetch_table_features('poi_features')
                    fetch_table_features('road_features')
                    fetch_table_features('rental_features')
                    fetch_table_features('demographic_features')
                    fetch_table_features('footfall_features')
                    fetch_table_features('transit_features')
                    fetch_table_features('income_features')
                    
                    # Fill defaults
                    for key, value in features.items():
                        if value is None:
                            features[key] = 0
                            
                    results.append(features)
                    
            return results
                
        finally:
            conn.close()
