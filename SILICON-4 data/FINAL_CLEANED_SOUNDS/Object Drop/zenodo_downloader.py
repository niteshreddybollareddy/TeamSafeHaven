#!/usr/bin/env python3
"""
ReaLISED Dataset Downloader (from Zenodo)

Downloads the Real-Life Indoor Sound Event Dataset which includes
"object falling" sounds among 18 event classes.

Dataset: https://zenodo.org/records/6488321
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess


OUTPUT_DIR = Path(__file__).parent
ZENODO_RECORD_ID = "6488321"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def get_download_urls():
    """Get download URLs from Zenodo API."""
    try:
        response = requests.get(ZENODO_API, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        files = data.get("files", [])
        return [(f["key"], f["links"]["self"]) for f in files]
    except Exception as e:
        print(f"Error accessing Zenodo API: {e}")
        return []


def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_and_process(zip_path, output_dir, target_duration=7):
    """Extract ZIP and process object falling sounds."""
    
    print("\n📦 Extracting files...")
    extract_dir = output_dir / "realised_temp"
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find object falling files
    all_audio = list(extract_dir.rglob("*.flac")) + list(extract_dir.rglob("*.wav"))
    
    # Filter for object falling (usually in folder name or filename)
    object_falling = [f for f in all_audio if 
                     "falling" in f.name.lower() or 
                     "falling" in str(f.parent).lower() or
                     "object" in f.name.lower() or
                     "drop" in f.name.lower() or
                     "impact" in f.name.lower()]
    
    if not object_falling:
        # If no specific files, use all
        object_falling = all_audio
    
    print(f"Found {len(object_falling)} relevant audio files")
    
    # Process to WAV
    processed = 0
    for i, audio_file in enumerate(object_falling):
        try:
            output_name = f"realised_{i+1:04d}.wav"
            output_path = output_dir / output_name
            
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-t", str(target_duration),
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            processed += 1
            
        except Exception as e:
            continue
    
    print(f"✅ Processed {processed} files")
    
    # Cleanup
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    return processed


def main():
    print("=" * 70)
    print("ReaLISED Dataset Downloader (Zenodo)")
    print("Real indoor sound events including object falling")
    print("=" * 70)
    
    # Get download URLs
    print("\n🔍 Fetching download information from Zenodo...")
    files = get_download_urls()
    
    if not files:
        print("\n⚠️ Could not fetch from Zenodo API.")
        print("Please download manually from: https://zenodo.org/records/6488321")
        return
    
    print(f"Found {len(files)} files to download")
    
    for filename, url in files:
        if filename.endswith(".zip"):
            zip_path = OUTPUT_DIR / filename
            
            print(f"\n📥 Downloading: {filename}")
            if download_file(url, zip_path, desc=filename):
                extract_and_process(zip_path, OUTPUT_DIR)
    
    print(f"\n✅ Download complete! Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
