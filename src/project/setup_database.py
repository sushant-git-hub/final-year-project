"""
Quick database setup script - Creates PostGIS extension if needed
Run this once before running feature_pipeline.py
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import psycopg2
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Create PostGIS extension in the database"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            port=os.getenv("DB_PORT", "5432")
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if PostGIS exists
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis');")
        exists = cursor.fetchone()[0]
        
        if not exists:
            print("Creating PostGIS extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            print("✓ PostGIS extension created")
        else:
            print("✓ PostGIS extension already exists")
        
        # Verify PostGIS version
        cursor.execute("SELECT PostGIS_version();")
        version = cursor.fetchone()[0]
        print(f"✓ PostGIS version: {version}")
        
        cursor.close()
        conn.close()
        print("\n✓ Database setup complete! You can now run feature_pipeline.py")
        
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database credentials are correct")
        print("3. PostGIS is installed on your system")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    setup_database()

