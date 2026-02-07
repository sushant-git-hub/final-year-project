"""
Check the results of the feature pipeline
"""

import psycopg2
from dotenv import load_dotenv
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def check_results():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            port=os.getenv("DB_PORT", "5432")
        )
        cursor = conn.cursor()
        
        print("=" * 60)
        print("MapMyStore - Data Processing Results")
        print("=" * 60)
        
        # Check grid cells
        cursor.execute("SELECT COUNT(*) FROM grid_cells;")
        grid_count = cursor.fetchone()[0]
        print(f"\nGrid Cells: {grid_count:,}")
        
        # Check POI features
        cursor.execute("SELECT COUNT(*) FROM poi_features;")
        poi_count = cursor.fetchone()[0]
        cursor.execute("""
            SELECT 
                COUNT(*) as total_cells,
                SUM(competitor_count) as total_competitors,
                AVG(competitor_count) as avg_competitors_per_cell,
                MAX(competitor_count) as max_competitors,
                AVG(nearest_m) as avg_nearest_distance
            FROM poi_features;
        """)
        poi_stats = cursor.fetchone()
        print(f"\nPOI Features: {poi_count:,} cells")
        print(f"  - Total competitors found: {poi_stats[1]:.0f}")
        print(f"  - Avg competitors per cell: {poi_stats[2]:.2f}")
        print(f"  - Max competitors in a cell: {poi_stats[3]}")
        print(f"  - Avg distance to nearest competitor: {poi_stats[4]:.2f} meters")
        
        # Check road features
        cursor.execute("SELECT COUNT(*) FROM road_features;")
        road_count = cursor.fetchone()[0]
        cursor.execute("""
            SELECT 
                COUNT(*) as total_cells,
                AVG(road_density_km) as avg_road_density,
                MAX(road_density_km) as max_road_density,
                AVG(major_dist_m) as avg_major_road_distance,
                MIN(major_dist_m) as min_major_road_distance
            FROM road_features;
        """)
        road_stats = cursor.fetchone()
        print(f"\nRoad Features: {road_count:,} cells")
        print(f"  - Avg road density: {road_stats[1]:.2f} meters/sq km")
        print(f"  - Max road density: {road_stats[2]:.2f} meters/sq km")
        print(f"  - Avg distance to major road: {road_stats[3]:.2f} meters")
        print(f"  - Min distance to major road: {road_stats[4]:.2f} meters")

        # Check rental features (if table exists)
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'rental_features'
            );
        """)
        has_rental = cursor.fetchone()[0]
        if has_rental:
            cursor.execute("SELECT COUNT(*) FROM rental_features;")
            rental_count = cursor.fetchone()[0]
            cursor.execute("""
                SELECT AVG(commercial_rent_per_sqft), MIN(commercial_rent_per_sqft), MAX(commercial_rent_per_sqft)
                FROM rental_features WHERE commercial_rent_per_sqft > 0;
            """)
            rental_stats = cursor.fetchone()
            print(f"\nRental Features: {rental_count:,} cells")
            if rental_stats[0]:
                print(f"  - Avg commercial rent: Rs {rental_stats[0]:.2f}/sqft")
                print(f"  - Min commercial rent: Rs {rental_stats[1]:.2f}/sqft")
                print(f"  - Max commercial rent: Rs {rental_stats[2]:.2f}/sqft")

        # Sample data
        print("\n" + "=" * 60)
        print("Sample Grid Cell Data:")
        print("=" * 60)
        if has_rental:
            cursor.execute("""
                SELECT 
                    g.cell_id, g.center_lat, g.center_lon,
                    p.competitor_count, p.nearest_m,
                    r.road_density_km, r.major_dist_m,
                    rt.commercial_rent_per_sqft, rt.zone
                FROM grid_cells g
                LEFT JOIN poi_features p ON g.cell_id = p.cell_id
                LEFT JOIN road_features r ON g.cell_id = r.cell_id
                LEFT JOIN rental_features rt ON g.cell_id = rt.cell_id
                WHERE p.competitor_count > 0
                ORDER BY p.competitor_count DESC
                LIMIT 5;
            """)
            print("\nTop 5 cells by competitor count:")
            print(f"{'Cell ID':<12} {'Lat':<10} {'Lon':<10} {'Competitors':<10} {'Nearest(m)':<10} {'Road Dens':<10} {'Major(m)':<10} {'Rent/sqft':<10} {'Zone':<12}")
            print("-" * 95)
            for row in cursor.fetchall():
                rent = f"{row[7]:.1f}" if row[7] else "-"
                zone = (row[8] or "-")[:12]
                print(f"{row[0]:<12} {row[1]:<10.6f} {row[2]:<10.6f} {row[3]:<10} {row[4]:<10.1f} {row[5]:<10.2f} {row[6]:<10.1f} {rent:<10} {zone:<12}")
        else:
            cursor.execute("""
                SELECT 
                    g.cell_id, g.center_lat, g.center_lon,
                    p.competitor_count, p.nearest_m,
                    r.road_density_km, r.major_dist_m
                FROM grid_cells g
                LEFT JOIN poi_features p ON g.cell_id = p.cell_id
                LEFT JOIN road_features r ON g.cell_id = r.cell_id
                WHERE p.competitor_count > 0
                ORDER BY p.competitor_count DESC
                LIMIT 5;
            """)
            print("\nTop 5 cells by competitor count:")
            print(f"{'Cell ID':<12} {'Lat':<10} {'Lon':<10} {'Competitors':<12} {'Nearest(m)':<12} {'Road Density':<15} {'Major Dist(m)':<15}")
            print("-" * 90)
            for row in cursor.fetchall():
                print(f"{row[0]:<12} {row[1]:<10.6f} {row[2]:<10.6f} {row[3]:<12} {row[4]:<12.1f} {row[5]:<15.2f} {row[6]:<15.1f}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("[OK] Data processing complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Collect demographic data (population, income)")
        print("2. Add more features (transit, footfall generators)")
        print("3. Build ML models for demand prediction")
        print("4. Build dashboard and API")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_results()

