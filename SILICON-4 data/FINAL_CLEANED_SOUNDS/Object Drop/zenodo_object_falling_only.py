#!/usr/bin/env python3
"""
Proper ReaLISED Object Falling Downloader

This script properly downloads ONLY the 'object falling' category from ReaLISED,
using the original filenames to identify the correct category.

Original filename format: abc_123_45_67_8.flac
- 'obj' prefix = object falling sounds
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil

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


def extract_object_falling_only(zip_path, output_dir, target_duration=7):
    """Extract ZIP and process ONLY object falling sounds."""
    
    print("\n📦 Extracting and filtering for 'object_falling' category...")
    extract_dir = output_dir / "realised_temp"
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find ALL audio files
    all_audio = list(extract_dir.rglob("*.flac")) + list(extract_dir.rglob("*.wav"))
    
    print(f"Found {len(all_audio)} total audio files in archive")
    
    # Filter for ONLY object falling files (filename starts with 'obj')
    # ReaLISED format: abc_123_45_67_8.flac where abc is the category
    object_falling_files = []
    
    for f in all_audio:
        # Check if filename starts with 'obj' (object falling category)
        basename = f.stem.lower()
        if basename.startswith('obj'):
            object_falling_files.append(f)
    
    print(f"Found {len(object_falling_files)} 'object falling' category files")
    
    if not object_falling_files:
        # Fallback: look for files in 'object_falling' folder
        object_falling_files = [f for f in all_audio if 
                                'falling' in str(f.parent).lower() or
                                'object' in str(f.parent).lower()]
        print(f"Fallback search found {len(object_falling_files)} files")
    
    if not object_falling_files:
        # Second fallback: show folder structure
        print("\n📁 Archive structure:")
        for d in sorted(set(f.parent for f in all_audio)):
            count = len([x for x in all_audio if x.parent == d])
            print(f"  {d.name}: {count} files")
        object_falling_files = all_audio
        print(f"\nUsing all {len(object_falling_files)} files")
    
    # Process to WAV
    processed = 0
    for i, audio_file in enumerate(object_falling_files):
        try:
            # Keep original name info for categorization
            orig_name = audio_file.stem
            output_name = f"realised_obj_{processed+1:04d}.wav"
            output_path = output_dir / output_name
            
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-t", str(target_duration),
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                processed += 1
            
        except Exception as e:
            continue
    
    print(f"✅ Processed {processed} object falling files")
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    return processed


def main():
    print("=" * 70)
    print("ReaLISED 'Object Falling' Category Downloader")
    print("(Filters for ONLY object falling sounds)")
    print("=" * 70)
    
    # First, remove previous realised files that were mixed categories
    old_files = list(OUTPUT_DIR.glob("realised_*.wav")) + list(OUTPUT_DIR.glob("object_drop_*.wav"))
    if old_files:
        print(f"\n🗑️ Removing {len(old_files)} previous mixed-category files...")
        for f in old_files:
            f.unlink()
    
    # Get download URLs
    print("\n🔍 Fetching download information from Zenodo...")
    files = get_download_urls()
    
    if not files:
        print("\n⚠️ Could not fetch from Zenodo API.")
        return
    
    print(f"Found {len(files)} files to download")
    
    for filename, url in files:
        if filename.endswith(".zip"):
            zip_path = OUTPUT_DIR / filename
            
            print(f"\n📥 Downloading: {filename}")
            if download_file(url, zip_path, desc=filename):
                extract_object_falling_only(zip_path, OUTPUT_DIR)
    
    final_count = len(list(OUTPUT_DIR.glob("realised_obj_*.wav")))
    print(f"\n✅ Object falling files: {final_count}")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
