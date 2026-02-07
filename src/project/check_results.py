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

        # Check demographic features (if table exists)
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'demographic_features'
            );
        """)
        has_demographic = cursor.fetchone()[0]
        if has_demographic:
            cursor.execute("SELECT COUNT(*) FROM demographic_features;")
            demo_count = cursor.fetchone()[0]
            cursor.execute("""
                SELECT AVG(total_population), MAX(total_population), MIN(total_population)
                FROM demographic_features WHERE total_population > 0;
            """)
            demo_stats = cursor.fetchone()
            print(f"\nDemographic Features: {demo_count:,} cells")
            if demo_stats[0]:
                print(f"  - Avg ward population per cell: {demo_stats[0]:,.0f}")
                print(f"  - Max ward population: {demo_stats[1]:,.0f}")
                print(f"  - Min ward population: {demo_stats[2]:,.0f}")

        # Sample data
        print("\n" + "=" * 60)
        print("Sample Grid Cell Data:")
        print("=" * 60)
        if has_rental or has_demographic:
            cols = ["g.cell_id", "g.center_lat", "g.center_lon", "p.competitor_count", "p.nearest_m", "r.road_density_km", "r.major_dist_m"]
            sel = ", ".join(cols)
            joins = "LEFT JOIN poi_features p ON g.cell_id = p.cell_id LEFT JOIN road_features r ON g.cell_id = r.cell_id"
            if has_rental:
                sel += ", rt.commercial_rent_per_sqft, rt.zone"
                joins += " LEFT JOIN rental_features rt ON g.cell_id = rt.cell_id"
            if has_demographic:
                sel += ", d.total_population, d.ward_name"
                joins += " LEFT JOIN demographic_features d ON g.cell_id = d.cell_id"
            cursor.execute(f"""
                SELECT {sel}
                FROM grid_cells g
                {joins}
                WHERE p.competitor_count > 0
                ORDER BY p.competitor_count DESC
                LIMIT 5;
            """)
            print("\nTop 5 cells by competitor count:")
            hdr = f"{'Cell ID':<12} {'Lat':<10} {'Lon':<10} {'Competitors':<10} {'Nearest(m)':<10} {'Road Dens':<10} {'Major(m)':<10}"
            if has_rental:
                hdr += f" {'Rent/sqft':<10} {'Zone':<12}"
            if has_demographic:
                hdr += f" {'Pop':<10} {'Ward':<15}"
            print(hdr)
            print("-" * (95 + (25 if has_demographic else 0)))
            for row in cursor.fetchall():
                line = f"{row[0]:<12} {row[1]:<10.6f} {row[2]:<10.6f} {row[3]:<10} {row[4]:<10.1f} {row[5]:<10.2f} {row[6]:<10.1f}"
                idx = 7
                if has_rental:
                    rent = f"{row[idx]:.1f}" if row[idx] else "-"
                    zone = (row[idx + 1] or "-")[:12]
                    line += f" {rent:<10} {zone:<12}"
                    idx += 2
                if has_demographic:
                    pop = f"{int(row[idx]):,}" if row[idx] is not None else "-"
                    ward = (str(row[idx + 1]) or "-")[:15] if idx + 1 < len(row) else "-"
                    line += f" {pop:<10} {ward:<15}"
                print(line)
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
        print("1. Add more features (transit, footfall generators)")
        print("2. Collect income data (if available)")
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

