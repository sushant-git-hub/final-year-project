# Automated Data Collection Scripts

This directory contains scripts to automatically collect missing data for the MapMyStore project.

## 📋 Available Scripts

### 1. **collect_footfall_generators.py**
Collects footfall generators (malls, IT parks, colleges, hospitals) from OpenStreetMap and adds known major locations.

**Output:** `data/raw/pune_footfall_generators.csv`

**Features:**
- Shopping malls
- IT parks and tech centers
- Educational institutions
- Hospitals
- Major landmarks

**Run:**
```powershell
python scripts/collect_footfall_generators.py
```

---

### 2. **collect_transit_data.py**
Collects public transit stops (bus stops, metro stations, railway stations) from OpenStreetMap.

**Output:** `data/raw/pune_transit_stops.csv`

**Features:**
- PMPML bus stops
- Pune Metro stations
- Railway stations
- Major bus terminals

**Run:**
```powershell
python scripts/collect_transit_data.py
```

---

### 3. **collect_income_proxy.py**
Generates income proxy data based on area characteristics and property prices.

**Output:** `data/raw/pune_income_proxy.csv`

**Features:**
- Income tier (High/Medium/Low)
- Average monthly income
- Property price per sqft
- Purchasing power index

**Run:**
```powershell
python scripts/collect_income_proxy.py
```

---

### 4. **generate_synthetic_labels.py**
Generates synthetic training labels for ML model demonstration.

**Output:** `data/raw/synthetic_training_labels.csv`

**Features:**
- Location score (0-100)
- Monthly revenue estimate
- Success label (binary)
- Daily footfall estimate
- Profitability category

**Requirements:** Database must be populated (run `feature_pipeline.py` first)

**Run:**
```powershell
python scripts/generate_synthetic_labels.py
```

---

### 5. **run_all_data_collection.py** (MASTER SCRIPT)
Runs all data collection scripts in sequence.

**Run:**
```powershell
python scripts/run_all_data_collection.py
```

---

## 🚀 Quick Start

### Option A: Run All Scripts at Once (RECOMMENDED)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master script
python scripts/run_all_data_collection.py
```

### Option B: Run Individual Scripts

```powershell
# Install dependencies first
pip install -r requirements.txt

# Then run scripts individually
python scripts/collect_footfall_generators.py
python scripts/collect_transit_data.py
python scripts/collect_income_proxy.py
python scripts/generate_synthetic_labels.py
```

---

## 📊 Expected Output

After running all scripts, you will have these new files in `data/raw/`:

| File | Description | Rows (approx) |
|------|-------------|---------------|
| `pune_footfall_generators.csv` | Malls, IT parks, colleges, hospitals | 50-100 |
| `pune_transit_stops.csv` | Bus stops, metro, railway stations | 100-500 |
| `pune_income_proxy.csv` | Income estimates by ward | 73 |
| `synthetic_training_labels.csv` | ML training labels | 1000+ |

---

## ⚙️ Configuration

### Google Places API (Optional)

To collect IT parks from Google Places API, add your API key to `.env`:

```env
GOOGLE_PLACES_API_KEY=your_api_key_here
```

**Note:** Scripts will work without Google API key (using OSM data only).

---

## 🔧 Dependencies

All dependencies are listed in `requirements.txt`:

- `requests` - HTTP requests for OSM Overpass API
- `pandas` - Data processing
- `numpy` - Numerical operations
- `psycopg2-binary` - PostgreSQL connection (for synthetic labels)
- `python-dotenv` - Environment variables

---

## 📝 Data Sources

| Data Type | Source | License |
|-----------|--------|---------|
| Footfall Generators | OpenStreetMap | ODbL |
| Transit Stops | OpenStreetMap | ODbL |
| Income Proxy | Derived from area characteristics | N/A |
| Synthetic Labels | Generated from existing features | N/A |

---

## ⏱️ Execution Time

| Script | Time | Network Required |
|--------|------|------------------|
| Footfall Generators | 2-3 min | Yes (OSM API) |
| Transit Data | 2-3 min | Yes (OSM API) |
| Income Proxy | 10 sec | No |
| Synthetic Labels | 30 sec | No (DB only) |
| **Total** | **5-7 min** | |

---

## 🐛 Troubleshooting

### "No data collected" error
- **Cause:** Network issue or OSM API timeout
- **Solution:** Check internet connection, wait 1 minute, try again

### "Database connection error"
- **Cause:** PostgreSQL not running or wrong credentials
- **Solution:** Check `.env` file, ensure PostgreSQL is running

### "Ward data not found"
- **Cause:** Missing `pune_wards_for_postgis.csv`
- **Solution:** Run `python scripts/Ward_demographic_cleaning.py` first

### OSM API rate limiting
- **Cause:** Too many requests to OSM
- **Solution:** Scripts have built-in delays, just wait and retry

---

## 🎯 Next Steps After Data Collection

1. **Update Feature Pipeline**
   ```powershell
   python src/project/feature_pipeline.py
   ```

2. **Verify Results**
   ```powershell
   python src/project/check_results.py
   ```

3. **Build ML Model** (coming soon)

4. **Build Backend API** (coming soon)

---

## 📄 Output File Formats

### pune_footfall_generators.csv
```csv
name,type,latitude,longitude,importance,source
Phoenix Marketcity,Mall,18.5593,73.7772,5,Manual
Rajiv Gandhi Infotech Park,IT Park,18.5913,73.7371,5,Manual
COEP,College,18.5287,73.8673,4,Manual
```

### pune_transit_stops.csv
```csv
name,type,latitude,longitude,operator,source
Pune Junction,Railway Station,18.5284,73.8742,Indian Railways,Manual
Swargate Bus Stand,Bus Terminal,18.5018,73.8636,PMPML,Manual
```

### pune_income_proxy.csv
```csv
ward_id,ward_name,income_tier,avg_monthly_income,property_price_sqft,purchasing_power_index
PMC_1,Aundh,High,75000,8000,150.5
PMC_2,Kothrud,Medium,45000,5000,90.2
```

### synthetic_training_labels.csv
```csv
cell_id,center_lat,center_lon,location_score,monthly_revenue,success_label,daily_footfall,profitability
cell_0,18.4025,73.7025,65,450000,1,180,Medium
cell_1,18.4025,73.7075,42,280000,0,95,Low
```

---

## ⚠️ Important Notes

1. **Synthetic Labels:** The training labels are SYNTHETIC (generated from features). For production, collect real store performance data.

2. **Income Data:** Income estimates are based on area characteristics, not actual census data. For better accuracy, use Census of India data.

3. **OSM Data Quality:** OpenStreetMap data quality varies by area. Some locations may be missing or outdated.

4. **API Costs:** Google Places API is optional and has costs. Scripts work fine without it using OSM data only.

---

## 📞 Support

If you encounter issues:
1. Check the error message
2. Verify internet connection
3. Ensure all dependencies are installed
4. Check database is running (for synthetic labels)
5. Review this README for troubleshooting steps
