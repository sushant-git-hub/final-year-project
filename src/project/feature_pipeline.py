"""
Feature pipeline for MapMyStore (Pune)
--------------------------------------
Creates a spatial grid, cleans POI and road data, computes baseline features,
and writes results to Postgres/PostGIS.

Requirements (pip):
  geopandas shapely pyproj pandas numpy psycopg2-binary python-dotenv

Usage:
  python feature_pipeline.py
"""

import os
import sys
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from shapely import wkt
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")

GRID_SIZE_M = 500  # grid resolution in meters
BOUNDS = {  # Pune approximate bounds
    "min_lat": 18.4,
    "max_lat": 18.65,
    "min_lon": 73.7,
    "max_lon": 73.95,
}

POI_CSV = os.path.join(_DATA_RAW, "pune_all_retail_stores.csv")
ROADS_CSV = os.path.join(_DATA_RAW, "pune_roads_data.csv")
LOCALITIES_CSV = os.path.join(_DATA_RAW, "pune_localities_for_postgis.csv")
WARDS_CSV = os.path.join(_DATA_RAW, "pune_wards_for_postgis.csv")
# New datasets
FOOTFALL_CSV = os.path.join(_DATA_RAW, "pune_footfall_generators.csv")
TRANSIT_CSV = os.path.join(_DATA_RAW, "pune_transit_stops.csv")
INCOME_CSV = os.path.join(_DATA_RAW, "pune_income_proxy.csv")
LABELS_CSV = os.path.join(_DATA_RAW, "real_training_labels.csv")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_env():
    load_dotenv()
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "db": os.getenv("DB_NAME", "postgres"),  # Default to postgres
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "port": os.getenv("DB_PORT", "5432"),
    }


def get_engine(cfg):
    url = (
        f"postgresql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['db']}"
    )
    return create_engine(url)


def create_grid(bounds, size_m):
    # Work in a projected CRS for Pune (Web Mercator is fine for this scale)
    xmin, ymin, xmax, ymax = (
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    )
    # Build in lat/lon then project to 3857 for metrics
    bbox = box(xmin, ymin, xmax, ymax)
    gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326").to_crs(3857)
    minx, miny, maxx, maxy = gdf.total_bounds

    rows = int(math.ceil((maxy - miny) / size_m))
    cols = int(math.ceil((maxx - minx) / size_m))

    cells = []
    cell_ids = []
    cid = 0
    for i in range(cols):
        for j in range(rows):
            x1 = minx + i * size_m
            y1 = miny + j * size_m
            x2 = x1 + size_m
            y2 = y1 + size_m
            cells.append(box(x1, y1, x2, y2))
            cell_ids.append(f"cell_{cid}")
            cid += 1

    grid = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": cells}, crs="EPSG:3857")
    # Calculate centroid in projected CRS (3857), then convert to lat/lon
    centroids_3857 = grid.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_3857, crs="EPSG:3857").to_crs(4326)
    grid["center_lon"] = centroids_4326.x
    grid["center_lat"] = centroids_4326.y
    return grid


def clean_poi(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.drop_duplicates(subset=["place_id"])
    df = df[df["business_status"].fillna("").str.upper() == "OPERATIONAL"]

    # valid coords in Pune bounds
    df = df[
        (df["latitude"].between(BOUNDS["min_lat"], BOUNDS["max_lat"]))
        & (df["longitude"].between(BOUNDS["min_lon"], BOUNDS["max_lon"]))
    ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3857)
    return gdf


def clean_roads(path):
    df = pd.read_csv(path, low_memory=False)
    records = []
    for _, row in df.iterrows():
        geom_wkt = row.get("geometry_wkt")
        geom = None
        if isinstance(geom_wkt, str) and geom_wkt.strip():
            try:
                geom = wkt.loads(geom_wkt)
            except Exception:
                geom = None
        if geom is None:
            slon, slat, elon, elat = (
                row.get("start_lon"),
                row.get("start_lat"),
                row.get("end_lon"),
                row.get("end_lat"),
            )
            if pd.notna(slon) and pd.notna(slat) and pd.notna(elon) and pd.notna(elat):
                geom = LineString([(slon, slat), (elon, elat)])
        if geom is None or geom.is_empty:
            continue
        records.append(row.to_dict() | {"geometry": geom})
    if not records:
        return gpd.GeoDataFrame(columns=df.columns.tolist() + ["geometry"])
    roads = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(3857)
    roads = roads.drop_duplicates(subset=["osmid", "geometry"])
    return roads


def compute_poi_features(grid, pois):
    # spatial join to count competitors within grid cell
    # count competitors: types contains supermarket/grocery/convenience
    competitors = pois[
        pois["types"].fillna("").str.contains(
            "supermarket|grocery_or_supermarket|convenience_store|store",
            case=False,
            regex=True,
        )
    ]
    joined = gpd.sjoin(grid, competitors, predicate="contains", how="left")
    counts = joined.groupby("cell_id").size().rename("competitor_count").reset_index()
    nearest = (
        gpd.sjoin_nearest(grid, competitors[["geometry"]], how="left", distance_col="nearest_m")
        .groupby("cell_id")["nearest_m"]
        .min()
        .reset_index()
    )
    features = grid[["cell_id"]].merge(counts, on="cell_id", how="left").merge(
        nearest, on="cell_id", how="left"
    )
    features["competitor_count"] = features["competitor_count"].fillna(0).astype(int)
    return features


def load_localities(path):
    """Load locality rental data and return GeoDataFrame (EPSG:3857)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if df.empty:
        return None
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3857)
    return gdf


def compute_rental_features(grid, localities):
    """Assign nearest locality's rental data to each grid cell."""
    if localities is None or localities.empty:
        return None
    cols = ["commercial_rent_per_sqft", "zone", "tier", "confidence", "locality"]
    cols = [c for c in cols if c in localities.columns]
    if not cols:
        return None
    nearest = gpd.sjoin_nearest(
        grid[["cell_id", "geometry"]],
        localities[["geometry"] + cols],
        how="left",
        distance_col="locality_dist_m",
    )
    out_cols = ["cell_id"] + cols + ["locality_dist_m"]
    feat = nearest[out_cols].drop_duplicates(subset="cell_id", keep="first")
    feat["commercial_rent_per_sqft"] = feat["commercial_rent_per_sqft"].fillna(0)
    return feat


def load_wards(path):
    """Load ward demographic data and return GeoDataFrame (EPSG:3857)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if df.empty:
        return None
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3857)
    return gdf


def load_footfall_generators(path):
    """Load footfall generator data (malls, IT parks, colleges, hospitals)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if df.empty:
        return None
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3857)
    return gdf


def load_transit_stops(path):
    """Load transit stop data (bus, metro, railway)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if df.empty:
        return None
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3857)
    return gdf


def load_income_proxy(path):
    """Load income proxy data by ward."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False)


def load_training_labels(path):
    """Load real training labels from store performance data."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if df.empty:
        return None
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3857)
    return gdf



def compute_demographic_features(grid, wards):
    """Assign nearest ward's population to each grid cell."""
    if wards is None or wards.empty:
        return None
    cols = ["total_population", "ward_id", "ward_name", "zone"]
    cols = [c for c in cols if c in wards.columns]
    if not cols:
        return None
    nearest = gpd.sjoin_nearest(
        grid[["cell_id", "geometry"]],
        wards[["geometry"] + cols],
        how="left",
        distance_col="ward_dist_m",
    )
    out_cols = ["cell_id"] + cols + ["ward_dist_m"]
    feat = nearest[out_cols].drop_duplicates(subset="cell_id", keep="first")
    feat["total_population"] = feat["total_population"].fillna(0).astype(int)
    return feat


def compute_road_features(grid, roads):
    # road density per cell (meters per sq km), major-road distance
    grid_proj = grid.copy()
    # Density
    overlay = gpd.overlay(roads[["geometry", "highway"]], grid_proj, how="intersection")
    overlay["len_m"] = overlay.geometry.length
    density = overlay.groupby("cell_id")["len_m"].sum().reset_index()
    cell_area_km2 = (GRID_SIZE_M * GRID_SIZE_M) / 1_000_000
    density["road_density_km"] = density["len_m"] / cell_area_km2

    # Major road distance (motorway/trunk/primary)
    majors = roads[
        roads["highway"].fillna("").str.contains("motorway|trunk|primary", case=False, regex=True)
    ]
    nearest = (
        gpd.sjoin_nearest(grid_proj, majors[["geometry"]], how="left", distance_col="major_dist_m")
        .groupby("cell_id")["major_dist_m"]
        .min()
        .reset_index()
    )

    features = grid_proj[["cell_id"]].merge(density[["cell_id", "road_density_km"]], on="cell_id", how="left")
    features = features.merge(nearest, on="cell_id", how="left")
    features["road_density_km"] = features["road_density_km"].fillna(0)
    return features


def compute_footfall_features(grid, footfall_gdf):
    """Compute footfall generator features for each grid cell."""
    if footfall_gdf is None or footfall_gdf.empty:
        return pd.DataFrame({"cell_id": grid["cell_id"], "footfall_generator_count": 0})
    
    # Count generators in each cell
    joined = gpd.sjoin(grid, footfall_gdf, predicate="contains", how="left")
    counts = joined.groupby("cell_id").size().rename("footfall_generator_count").reset_index()
    
    # Count by type
    type_counts = joined.groupby(["cell_id", "type"]).size().unstack(fill_value=0).reset_index()
    type_counts.columns = ["cell_id"] + [f"{col.lower().replace(' ', '_')}_count" for col in type_counts.columns[1:]]
    
    # Distance to nearest generator
    nearest = (
        gpd.sjoin_nearest(grid, footfall_gdf[["geometry"]], how="left", distance_col="nearest_generator_m")
        .groupby("cell_id")["nearest_generator_m"]
        .min()
        .reset_index()
    )
    
    # Merge all features
    features = grid[["cell_id"]].merge(counts, on="cell_id", how="left")
    features = features.merge(type_counts, on="cell_id", how="left")
    features = features.merge(nearest, on="cell_id", how="left")
    features["footfall_generator_count"] = features["footfall_generator_count"].fillna(0).astype(int)
    
    return features


def compute_transit_features(grid, transit_gdf):
    """Compute transit accessibility features for each grid cell."""
    if transit_gdf is None or transit_gdf.empty:
        return pd.DataFrame({"cell_id": grid["cell_id"], "transit_stop_count": 0})
    
    # Count transit stops in each cell
    joined = gpd.sjoin(grid, transit_gdf, predicate="contains", how="left")
    counts = joined.groupby("cell_id").size().rename("transit_stop_count").reset_index()
    
    # Count by type
    type_counts = joined.groupby(["cell_id", "type"]).size().unstack(fill_value=0).reset_index()
    type_counts.columns = ["cell_id"] + [f"{col.lower().replace(' ', '_')}_count" for col in type_counts.columns[1:]]
    
    # Distance to nearest transit stop
    nearest = (
        gpd.sjoin_nearest(grid, transit_gdf[["geometry"]], how="left", distance_col="nearest_transit_m")
        .groupby("cell_id")["nearest_transit_m"]
        .min()
        .reset_index()
    )
    
    # Merge all features
    features = grid[["cell_id"]].merge(counts, on="cell_id", how="left")
    features = features.merge(type_counts, on="cell_id", how="left")
    features = features.merge(nearest, on="cell_id", how="left")
    features["transit_stop_count"] = features["transit_stop_count"].fillna(0).astype(int)
    
    return features


def compute_income_features(grid, wards_gdf, income_df):
    """Compute income features by joining ward income data with grid cells."""
    if wards_gdf is None or income_df is None:
        return pd.DataFrame({"cell_id": grid["cell_id"]})
    
    # Merge income data with ward geometries
    wards_with_income = wards_gdf.merge(income_df, on="ward_name", how="left")
    
    # Spatial join with grid
    joined = gpd.sjoin(grid, wards_with_income, how="left", predicate="intersects")
    
    # Aggregate by cell_id
    income_features = joined.groupby("cell_id").agg({
        "avg_monthly_income": "mean",
        "property_price_sqft": "mean",
        "purchasing_power_index": "mean",
        "income_tier": lambda x: x.mode()[0] if len(x) > 0 and x.notna().any() else None
    }).reset_index()
    
    return income_features



def write_to_db(engine, grid, poi_feat, road_feat, rental_feat=None, demographic_feat=None,
                footfall_feat=None, transit_feat=None, income_feat=None, labels=None):
    # Write grid and features
    grid_out = grid[["cell_id", "center_lon", "center_lat", "geometry"]].to_crs(4326)
    grid_out.to_postgis("grid_cells", engine, if_exists="replace", index=False)

    # Merge features back with grid geometry for PostGIS
    poi_gdf = grid[["cell_id", "geometry"]].merge(poi_feat, on="cell_id", how="left")
    poi_gdf = poi_gdf[["cell_id"] + [c for c in poi_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
    poi_gdf.to_postgis("poi_features", engine, if_exists="replace", index=False)

    road_gdf = grid[["cell_id", "geometry"]].merge(road_feat, on="cell_id", how="left")
    road_gdf = road_gdf[["cell_id"] + [c for c in road_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
    road_gdf.to_postgis("road_features", engine, if_exists="replace", index=False)

    tables = ["grid_cells", "poi_features", "road_features"]

    if rental_feat is not None:
        rental_gdf = grid[["cell_id", "geometry"]].merge(rental_feat, on="cell_id", how="left")
        rental_gdf = rental_gdf[["cell_id"] + [c for c in rental_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
        rental_gdf.to_postgis("rental_features", engine, if_exists="replace", index=False)
        tables.append("rental_features")

    if demographic_feat is not None:
        demo_gdf = grid[["cell_id", "geometry"]].merge(demographic_feat, on="cell_id", how="left")
        demo_gdf = demo_gdf[["cell_id"] + [c for c in demographic_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
        demo_gdf.to_postgis("demographic_features", engine, if_exists="replace", index=False)
        tables.append("demographic_features")

    # NEW - Add footfall features
    if footfall_feat is not None:
        footfall_gdf = grid[["cell_id", "geometry"]].merge(footfall_feat, on="cell_id", how="left")
        footfall_gdf = footfall_gdf[["cell_id"] + [c for c in footfall_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
        footfall_gdf.to_postgis("footfall_features", engine, if_exists="replace", index=False)
        tables.append("footfall_features")

    # NEW - Add transit features
    if transit_feat is not None:
        transit_gdf = grid[["cell_id", "geometry"]].merge(transit_feat, on="cell_id", how="left")
        transit_gdf = transit_gdf[["cell_id"] + [c for c in transit_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
        transit_gdf.to_postgis("transit_features", engine, if_exists="replace", index=False)
        tables.append("transit_features")

    # NEW - Add income features
    if income_feat is not None:
        income_gdf = grid[["cell_id", "geometry"]].merge(income_feat, on="cell_id", how="left")
        income_gdf = income_gdf[["cell_id"] + [c for c in income_feat.columns if c != "cell_id"] + ["geometry"]].to_crs(4326)
        income_gdf.to_postgis("income_features", engine, if_exists="replace", index=False)
        tables.append("income_features")

    # NEW - Add training labels (as separate table, not grid-based)
    if labels is not None:
        labels_gdf = labels.to_crs(4326)
        labels_gdf.to_postgis("training_labels", engine, if_exists="replace", index=False)
        tables.append("training_labels")

    print(f"[OK] Written {', '.join(tables)} to Postgres")



def main():
    cfg = load_env()
    print("Loading data...")
    pois = clean_poi(POI_CSV)
    roads = clean_roads(ROADS_CSV)
    print(f"POIs kept: {len(pois)}, Roads kept: {len(roads)}")

    localities = load_localities(LOCALITIES_CSV)
    if localities is not None:
        print(f"Localities loaded: {len(localities)}")
    else:
        print("Locality rental data not found (skipping rental features)")

    wards = load_wards(WARDS_CSV)
    if wards is not None:
        print(f"Wards loaded: {len(wards)}")
    else:
        print("Ward demographic data not found (skipping demographic features)")

    print("Building grid...")
    grid = create_grid(BOUNDS, GRID_SIZE_M)
    print(f"Grid cells: {len(grid)}")

    print("Computing POI features...")
    poi_feat = compute_poi_features(grid, pois)
    print("Computing road features...")
    road_feat = compute_road_features(grid, roads)

    rental_feat = None
    if localities is not None:
        print("Computing rental features...")
        rental_feat = compute_rental_features(grid, localities)

    demographic_feat = None
    if wards is not None:
        print("Computing demographic features...")
        demographic_feat = compute_demographic_features(grid, wards)

    # NEW - Load footfall generators
    footfall = load_footfall_generators(FOOTFALL_CSV)
    if footfall is not None:
        print(f"Footfall generators loaded: {len(footfall)}")
    else:
        print("Footfall generator data not found (skipping footfall features)")

    # NEW - Load transit stops
    transit = load_transit_stops(TRANSIT_CSV)
    if transit is not None:
        print(f"Transit stops loaded: {len(transit)}")
    else:
        print("Transit data not found (skipping transit features)")

    # NEW - Load income proxy
    income = load_income_proxy(INCOME_CSV)
    if income is not None:
        print(f"Income proxy data loaded: {len(income)} wards")
    else:
        print("Income proxy data not found (skipping income features)")

    # NEW - Load training labels
    labels = load_training_labels(LABELS_CSV)
    if labels is not None:
        print(f"Training labels loaded: {len(labels)} stores")
    else:
        print("Training labels not found (skipping)")

    # NEW - Compute footfall features
    footfall_feat = None
    if footfall is not None:
        print("Computing footfall features...")
        footfall_feat = compute_footfall_features(grid, footfall)

    # NEW - Compute transit features
    transit_feat = None
    if transit is not None:
        print("Computing transit features...")
        transit_feat = compute_transit_features(grid, transit)

    # NEW - Compute income features
    income_feat = None
    if income is not None and wards is not None:
        print("Computing income features...")
        income_feat = compute_income_features(grid, wards, income)

    print("Writing to Postgres...")
    engine = get_engine(cfg)
    write_to_db(engine, grid, poi_feat, road_feat, rental_feat, demographic_feat,
                footfall_feat, transit_feat, income_feat, labels)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

