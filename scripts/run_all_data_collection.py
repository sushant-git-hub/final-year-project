"""
Master script to run all data collection scripts
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    print("=" * 70)
    print("MapMyStore - Automated Data Collection")
    print("=" * 70)
    print("\nThis will collect:")
    print("  1. Footfall generators (malls, IT parks, colleges, hospitals)")
    print("  2. Transit accessibility (bus stops, metro, railway)")
    print("  3. Income proxy data")
    print("  4. Synthetic training labels")
    print("\n⏱️  Estimated time: 5-10 minutes")
    print("=" * 70)
    
    proceed = input("\nProceed with data collection? (yes/no): ").lower()
    if proceed != 'yes':
        print("Cancelled.")
        return
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        ("collect_footfall_generators.py", "Footfall Generators Collection"),
        ("collect_transit_data.py", "Transit Accessibility Collection"),
        ("collect_income_proxy.py", "Income Proxy Data Generation"),
        ("generate_synthetic_labels.py", "Synthetic Training Labels Generation"),
    ]
    
    results = []
    for script_name, description in scripts:
        script_path = os.path.join(scripts_dir, script_name)
        success = run_script(script_path, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("Data Collection Summary")
    print("=" * 70)
    
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\n{successful}/{len(results)} scripts completed successfully")
    
    if successful == len(results):
        print("\n🎉 All data collection completed!")
        print("\nNext steps:")
        print("  1. Run: python src/project/feature_pipeline.py")
        print("  2. Verify: python src/project/check_results.py")
        print("  3. Build backend API")
    else:
        print("\n⚠️  Some scripts failed. Check the errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
