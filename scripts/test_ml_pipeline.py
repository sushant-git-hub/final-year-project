"""
Quick Test Script for ML Pipeline
----------------------------------
Verify that all components are properly installed and configured.

Usage:
    python scripts/test_ml_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required packages are installed."""
    print("🔍 Testing package imports...")
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'geopandas': 'geopandas',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sqlalchemy': 'sqlalchemy',
        'psycopg2': 'psycopg2-binary',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All packages installed!")
    return True


def test_project_structure():
    """Test that all required files exist."""
    print("\n🔍 Testing project structure...")
    
    required_files = [
        'src/project/feature_engineering.py',
        'models/model_config.py',
        'models/train_success_model.py',
        'models/train_revenue_model.py',
        'scripts/prepare_training_data.py',
        'scripts/predict_locations.py',
        'requirements_ml.txt',
        'models/README.md'
    ]
    
    missing = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing files: {len(missing)}")
        return False
    
    print("\n✅ All files present!")
    return True


def test_database_connection():
    """Test database connection."""
    print("\n🔍 Testing database connection...")
    
    try:
        from dotenv import load_dotenv
        import os
        from sqlalchemy import create_engine
        
        load_dotenv()
        
        cfg = {
            "host": os.getenv("DB_HOST", "localhost"),
            "db": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "port": os.getenv("DB_PORT", "5432"),
        }
        
        url = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['db']}"
        engine = create_engine(url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        
        print(f"   ✓ Connected to {cfg['db']} at {cfg['host']}:{cfg['port']}")
        print("\n✅ Database connection successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("\n⚠️  Make sure PostgreSQL is running and .env is configured")
        return False


def test_data_availability():
    """Test that required data files exist."""
    print("\n🔍 Testing data availability...")
    
    data_files = [
        'data/raw/real_training_labels.csv',
        'data/raw/pune_all_retail_stores.csv',
        'data/raw/pune_roads_data.csv',
        'data/raw/pune_localities_for_postgis.csv',
        'data/raw/pune_wards_for_postgis.csv',
        'data/raw/pune_footfall_generators.csv',
        'data/raw/pune_transit_stops.csv',
        'data/raw/pune_income_proxy.csv'
    ]
    
    missing = []
    for file_path in data_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ⚠️  {file_path} - NOT FOUND")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Some data files missing (this is OK if you haven't run data collection yet)")
    else:
        print("\n✅ All data files present!")
    
    return True


def test_feature_engineering():
    """Test feature engineering functions."""
    print("\n🔍 Testing feature engineering...")
    
    try:
        from src.project.feature_engineering import (
            add_city_center_distance,
            add_interaction_features,
            handle_missing_values,
            get_feature_columns
        )
        import pandas as pd
        
        # Create dummy data
        df = pd.DataFrame({
            'center_lat': [18.5, 18.6],
            'center_lon': [73.8, 73.9],
            'commercial_rent_per_sqft': [100, 150],
            'avg_monthly_income': [50000, 60000],
            'competitor_count': [5, 10],
            'total_population': [10000, 15000]
        })
        
        # Test functions
        df = add_city_center_distance(df)
        assert 'distance_to_city_center' in df.columns
        
        df = add_interaction_features(df)
        assert 'rent_to_income_ratio' in df.columns
        
        df = handle_missing_values(df)
        
        feature_cols = get_feature_columns()
        assert len(feature_cols) > 0
        
        print("   ✓ Feature engineering functions work correctly")
        print("\n✅ Feature engineering test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Feature engineering test failed: {e}")
        return False


def main():
    print("=" * 70)
    print("ML PIPELINE VERIFICATION TEST")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Database Connection", test_database_connection()))
    results.append(("Data Availability", test_data_availability()))
    results.append(("Feature Engineering", test_feature_engineering()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("1. Run: python scripts/prepare_training_data.py")
        print("2. Run: python models/train_success_model.py")
        print("3. Run: python models/train_revenue_model.py")
        print("4. Run: python scripts/predict_locations.py")
        print("=" * 70)
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as exc:
        print(f"\n\n❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
