#!/usr/bin/env python3
"""
Download model files from GitHub Releases
"""
import os
import requests
from pathlib import Path

GITHUB_USER = "shubhamkataria2005"
REPO_NAME = "ai-chatbot-mlService"
RELEASE_TAG = "v1.0.0"
BASE_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"

# Files to download
FILES = {
    "car_recognition/models/car_model.h5": "car_model.h5",
    "tradein_model/models/tradein_model.pkl": "tradein_model.pkl",
    "tradein_model/models/encoder_body_type.pkl": "encoder_body_type.pkl",
    "tradein_model/models/encoder_condition.pkl": "encoder_condition.pkl",
    "tradein_model/models/encoder_fuel_type.pkl": "encoder_fuel_type.pkl",
    "tradein_model/models/encoder_make.pkl": "encoder_make.pkl",
    "tradein_model/models/encoder_model.pkl": "encoder_model.pkl",
    "tradein_model/models/encoder_transmission.pkl": "encoder_transmission.pkl",
    "tradein_model/models/features.pkl": "features.pkl",
    "tradein_model/models/numerical_cols.pkl": "numerical_cols.pkl",
    "tradein_model/models/scaler.pkl": "scaler.pkl",
}

def download_file(url, dest_path):
    """Download a file with progress bar"""
    try:
        print(f"Downloading: {dest_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r  {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end='')
        print(f"\n✅ Downloaded: {dest_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {dest_path}: {e}")
        return False

def main():
    print("=" * 60)
    print("Downloading ML Models from GitHub Releases")
    print("=" * 60)
    print(f"Release: {RELEASE_TAG}")
    print("=" * 60)
    
    missing = []
    for dest_path in FILES.keys():
        if not os.path.exists(dest_path):
            missing.append(dest_path)
    
    if not missing:
        print("\n✅ All models already exist!")
        return 0
    
    print(f"\n📥 Downloading {len(missing)} files...")
    for dest_path in missing:
        filename = FILES[dest_path]
        url = f"{BASE_URL}/{filename}"
        if not download_file(url, dest_path):
            return 1
    
    print("\n✅ All models downloaded successfully!")
    return 0

if __name__ == "__main__":
    exit(main())