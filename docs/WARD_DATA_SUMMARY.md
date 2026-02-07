# Pune Ward Demographic Data - Summary

## Source: `pune_ward_reservation.csv`

### What the file contains

| Column | Description | Status |
|--------|-------------|--------|
| **Ward** | Ward number (1-41 PMC, 1-32 PCMC) | ✓ |
| **Total_Population** | Ward-level population count | ✓ |
| **Ward_Name** | Ward name (e.g. Aundh, Baner, Kothrud) | ✓ |
| **Latitude, Longitude** | Ward centroid coordinates | ✓ |

### Coverage

- **PMC (Pune Municipal Corporation):** 41 wards
- **PCMC (Pimpri-Chinchwad):** 32 wards
- **Total:** 73 wards

### Per BE Report – demographic data needs

| Required | In ward file? | Notes |
|----------|----------------|-------|
| **Population density** | ✓ (as total_population) | Ward-level; used as proxy for demand |
| **Income levels** | ✗ | Not in this file; would need Census/other source |
| **Household characteristics** | ✗ | Not in this file |

---

## Preprocessing applied

1. **Empty rows:** Dropped (separator between PMC and PCMC).
2. **Zone:** Assigned `PMC` or `PCMC` from block structure.
3. **Unique `ward_id`:** Added (e.g. `PMC_1`, `PCMC_1`) to avoid duplicate ward numbers.
4. **Encoding:** Handled non-UTF-8 characters (e.g. `–` in Ajmera–Morwadi).
5. **Output:** `pune_wards_for_postgis.csv` in `data/raw/`.

---

## Pipeline integration

- **Preprocessing:** `python scripts/Ward_demographic_cleaning.py`
- **Feature pipeline:** Loads wards, uses nearest-ward spatial join; each grid cell gets `total_population`, `ward_id`, `ward_name`, `zone`.
- **PostGIS table:** `demographic_features`

---

## Data not in this file

| Data | Use case | Possible sources |
|------|----------|------------------|
| **Income** | Demand prediction | Census of India, market surveys |
| **Household size** | Demand prediction | Census |
| **Ward boundaries** | Better spatial join | Municipal/GIS shapefiles |
| **Population density (per sq km)** | Finer metric | Ward area + population |

---

## How to run

```powershell
# 1. Clean ward data
python scripts/Ward_demographic_cleaning.py

# 2. Run full pipeline (includes demographic features)
python src/project/feature_pipeline.py

# 3. Verify
python src/project/check_results.py
```
