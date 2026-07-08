#!/usr/bin/env python3
"""
validate_setup.py - Validate that all dependencies and configurations are correct
"""

import sys
import os

def check_env_file():
    """Check if .env file exists with required keys"""
    if not os.path.exists('.env'):
        print(".env file not found!")
        return False

    with open('.env', 'r') as f:
        content = f.read()

    required_keys = ['AQICN_TOKEN', 'HOPSWORKS_API_KEY']
    missing = [key for key in required_keys if key not in content]

    if missing:
        print(f"Missing environment variables in .env: {', '.join(missing)}")
        return False

    print(".env file configured correctly")
    return True

def check_directories():
    """Check if required directories exist"""
    dirs = ['data', 'data/raw', 'models', 'pipelines', 'notebooks']
    missing = [d for d in dirs if not os.path.exists(d)]

    if missing:
        print(f"Missing directories: {', '.join(missing)}")
        return False

    print("All required directories exist")
    return True

def check_files():
    """Check if required files exist"""
    files = [
        'pipelines/fetch_raw.py',
        'pipelines/feature_pipeline.py',
        'pipelines/training_pipeline.py',
        'pipelines/backfill.py',
        'requirements.txt'
    ]
    missing = [f for f in files if not os.path.exists(f)]

    if missing:
        print(f"Missing files: {', '.join(missing)}")
        return False

    print("All required files exist")
    return True

def check_imports():
    """Check if key imports work"""
    try:
        import pandas
        import numpy
        import sklearn
        import requests
        import hopsworks
        print("Core dependencies can be imported")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def main():
    print("\n" + "="*60)
    print("AQI PREDICTION PROJECT VALIDATION")
    print("="*60 + "\n")

    checks = [
        check_directories(),
        check_files(),
        check_env_file(),
        check_imports()
    ]

    print("\n" + "="*60)
    if all(checks):
        print("ALL CHECKS PASSED - Ready for GitHub Actions!")
        print("="*60 + "\n")
        return 0
    else:
        print("SOME CHECKS FAILED - Fix issues before deploying")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
