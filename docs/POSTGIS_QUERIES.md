# PostGIS Structure & Query Reference

Your project writes spatial data to a **PostgreSQL + PostGIS** database. Tables are created by `feature_pipeline.py` and `predict_locations.py`.

---

## 1. PostGIS Tables

| Table | Description |
|-------|-------------|
| `grid_cells` | Grid cells (cell_id, center_lat, center_lon, geometry) |
| `poi_features` | POI/competitor counts per cell |
| `road_features` | Road density per cell |
| `rental_features` | Rental/commercial rent per cell |
| `demographic_features` | Ward demographics per cell |
| `footfall_features` | Footfall generators per cell |
| `transit_features` | Transit stops per cell |
| `income_features` | Income/property price per cell |
| `training_labels` | Store labels for ML (rating, success_label, etc.) |
| `location_predictions` | ML predictions per cell (created by `predict_locations.py`) |

---

## 2. Connection (where to get credentials)

Credentials come from a **`.env`** file in the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
```

Use these same values in any tool (psql, pgAdmin, Python, backend).

---

## 3. Queries to Select Tables

### List all PostGIS tables

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

### Select each table (sample rows)

```sql
-- Grid
SELECT * FROM grid_cells LIMIT 10;

-- POI
SELECT * FROM poi_features LIMIT 10;

-- Roads
SELECT * FROM road_features LIMIT 10;

-- Rental
SELECT * FROM rental_features LIMIT 10;

-- Demographics
SELECT * FROM demographic_features LIMIT 10;

-- Footfall
SELECT * FROM footfall_features LIMIT 10;

-- Transit
SELECT * FROM transit_features LIMIT 10;

-- Income
SELECT * FROM income_features LIMIT 10;

-- Training labels
SELECT * FROM training_labels LIMIT 10;

-- Predictions (if you ran predict_locations.py)
SELECT * FROM location_predictions LIMIT 10;
```

### Joined features (all cells with all feature tables)

```sql
SELECT 
    g.cell_id,
    g.center_lat,
    g.center_lon,
    p.competitor_count,
    r.road_density_km,
    rf.commercial_rent_per_sqft,
    d.total_population,
    f.footfall_generator_count,
    t.transit_stop_count,
    i.avg_monthly_income
FROM grid_cells g
LEFT JOIN poi_features p ON g.cell_id = p.cell_id
LEFT JOIN road_features r ON g.cell_id = r.cell_id
LEFT JOIN rental_features rf ON g.cell_id = rf.cell_id
LEFT JOIN demographic_features d ON g.cell_id = d.cell_id
LEFT JOIN footfall_features f ON g.cell_id = f.cell_id
LEFT JOIN transit_features t ON g.cell_id = t.cell_id
LEFT JOIN income_features i ON g.cell_id = i.cell_id
LIMIT 100;
```

---

## 4. Where to Write or Run the Queries

### Option A: Command line (psql)

1. Open terminal in project root.
2. Run (replace with your `.env` values):

```bash
psql -h localhost -p 5432 -U postgres -d postgres
```

3. Paste any of the SQL above and press Enter.

### Option B: Python script (already in project)

- **`scripts/verify_new_features.py`** – Connects with `psycopg2`, runs queries, prints stats. Run:

```bash
python scripts/verify_new_features.py
```

- **`scripts/prepare_training_data.py`** – Uses SQLAlchemy and a big `SELECT` that joins all feature tables; run from project root:

```bash
python scripts/prepare_training_data.py
```

You can add new queries in:
- `scripts/verify_new_features.py` (after the existing `cur.execute(...)` blocks), or
- A new script that uses the same connection pattern (see `verify_new_features.py` or `prepare_training_data.py`).

### Option C: GUI (pgAdmin, DBeaver, etc.)

1. Create a connection with the same `.env` values (host, port, database, user, password).
2. Open a **Query** / **SQL** window.
3. Paste and run any of the SQL from section 3.

### Option D: Backend (Node.js)

- Config: `backend/src/config/db.config.js` (uses `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`).
- DB client: `backend/src/db/postgre/client.js` is currently commented out. To run raw SQL from the backend, uncomment that client and use it in your routes/services to execute the same queries.

---

## 5. Quick checklist

| Goal | What to do |
|------|------------|
| See all table names | Run the `information_schema` query in psql or pgAdmin. |
| Inspect one table | `SELECT * FROM <table_name> LIMIT 10` in psql/pgAdmin. |
| Verify pipeline output | Run `python scripts/verify_new_features.py`. |
| Get joined features for ML | Run `python scripts/prepare_training_data.py` or use the joined SQL in section 3. |

All queries use the **public** schema and the credentials from your project **`.env`** file.
