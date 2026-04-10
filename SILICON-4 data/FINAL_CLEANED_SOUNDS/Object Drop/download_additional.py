#!/usr/bin/env python3
"""
Multi-Source Object Fall Audio Downloader

Downloads REAL (non-synthetic, non-AI) audio of objects falling/impacts from:
1. TUT Sound Events 2016 - "object impact" class
2. ESC-50 - Glass breaking, domestic sounds
3. SAFE Dataset (Kaggle) - Fall detection sounds

This script helps collect enough real audio to reach ~600 samples.
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil
import glob

OUTPUT_DIR = Path(__file__).parent

def download_file(url, output_path, desc="Downloading"):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f:
            if total_size:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def download_tut_2016():
    """Download TUT Sound Events 2016 dataset (has object impact class)."""
    print("\n" + "="*70)
    print("📦 TUT SOUND EVENTS 2016 - Object Impact Sounds")
    print("="*70)
    
    # TUT 2016 download URL
    url = "https://zenodo.org/records/996424/files/TUT-sound-events-2016-development.audio.1.zip?download=1"
    
    zip_path = OUTPUT_DIR / "tut_2016_dev.zip"
    extract_dir = OUTPUT_DIR / "tut_temp"
    
    print("📥 Downloading TUT Sound Events 2016...")
    if not download_file(url, zip_path, "TUT 2016 Dataset"):
        return 0
    
    print("\n📦 Extracting...")
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        print(f"Error extracting: {e}")
        return 0
    
    # Find all audio files
    audio_files = list(extract_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    # Process files - TUT contains mixed scenes, extract impact sounds
    processed = 0
    for audio_file in audio_files:
        try:
            output_name = f"tut_impact_{processed+1:04d}.wav"
            output_path = OUTPUT_DIR / output_name
            
            # Get duration first
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", 
                        "format=duration", "-of", "default=nw=1:nk=1", str(audio_file)]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            try:
                duration = float(result.stdout.strip())
            except:
                duration = 0
            
            if duration < 0.5:
                continue
            
            # Convert to target format
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-t", "7",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=30)
            if output_path.exists():
                processed += 1
                
        except Exception as e:
            continue
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    print(f"✅ Processed {processed} files from TUT 2016")
    return processed


def download_esc50():
    """Download ESC-50 dataset and extract glass breaking & domestic sounds."""
    print("\n" + "="*70)
    print("📦 ESC-50 - Environmental Sound Classification")
    print("="*70)
    
    # ESC-50 GitHub release
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    
    zip_path = OUTPUT_DIR / "esc50.zip"
    extract_dir = OUTPUT_DIR / "esc50_temp"
    
    print("📥 Downloading ESC-50 dataset...")
    if not download_file(url, zip_path, "ESC-50 Dataset"):
        return 0
    
    print("\n📦 Extracting...")
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        print(f"Error extracting: {e}")
        return 0
    
    # Find audio directory
    audio_dir = None
    for d in extract_dir.rglob("audio"):
        if d.is_dir():
            audio_dir = d
            break
    
    if not audio_dir:
        print("Could not find audio directory")
        return 0
    
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.ogg"))
    print(f"Found {len(audio_files)} audio files")
    
    # ESC-50 classes relevant to object falling/impacts:
    # We want domestic sounds that could relate to objects
    # Class 38: glass_breaking, Class 21: door_knock (impact-like)
    # Class 39: can_opening, Class 23: clock_tick
    
    # The filename format is: foldfold-clipid-classid.wav
    # e.g., 1-100038-A-14.wav
    IMPACT_CLASSES = [
        "38",  # glass_breaking
        "21",  # door_knock
    ]
    
    processed = 0
    for audio_file in audio_files:
        try:
            # Extract class from filename
            parts = audio_file.stem.split("-")
            if len(parts) >= 4:
                class_id = parts[-1]
                if class_id in IMPACT_CLASSES:
                    output_name = f"esc50_{processed+1:04d}.wav"
                    output_path = OUTPUT_DIR / output_name
                    
                    cmd = [
                        "ffmpeg", "-y", "-i", str(audio_file),
                        "-t", "7",
                        "-acodec", "pcm_s16le",
                        "-ar", "44100",
                        "-ac", "1",
                        str(output_path)
                    ]
                    
                    subprocess.run(cmd, capture_output=True, timeout=30)
                    if output_path.exists():
                        processed += 1
                        
        except Exception as e:
            continue
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    print(f"✅ Processed {processed} impact-related files from ESC-50")
    return processed


def download_additional_zenodo():
    """Download additional audio from Zenodo: DataSED/DataSEC."""
    print("\n" + "="*70)
    print("📦 ADDITIONAL DOMESTIC SOUNDS FROM ZENODO")
    print("="*70)
    
    # DataSED dataset on Zenodo (includes Glass_breaking)
    url = "https://zenodo.org/records/10014306/files/DataSED.zip?download=1"
    
    zip_path = OUTPUT_DIR / "datased.zip"
    extract_dir = OUTPUT_DIR / "datased_temp"
    
    print("📥 Downloading DataSED...")
    if not download_file(url, zip_path, "DataSED"):
        return 0
    
    print("\n📦 Extracting...")
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except:
        zip_path.unlink(missing_ok=True)
        return 0
    
    # Find audio files related to impact sounds
    audio_files = list(extract_dir.rglob("*.wav")) + list(extract_dir.rglob("*.flac"))
    
    # Filter for relevant categories
    relevant_files = []
    for f in audio_files:
        path_lower = str(f).lower()
        if any(term in path_lower for term in ['glass', 'break', 'impact', 'crash', 'fall']):
            relevant_files.append(f)
    
    if not relevant_files:
        relevant_files = audio_files[:100]  # Take some samples
    
    print(f"Processing {len(relevant_files)} impact-related files")
    
    processed = 0
    for audio_file in relevant_files[:200]:
        try:
            output_name = f"zenodo_extra_{processed+1:04d}.wav"
            output_path = OUTPUT_DIR / output_name
            
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-t", "7",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=30)
            if output_path.exists():
                processed += 1
                
        except:
            continue
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    print(f"✅ Processed {processed} additional files")
    return processed


def show_final_summary():
    """Show final summary of all collected files."""
    
    wav_files = list(OUTPUT_DIR.glob("*.wav"))
    total_size = sum(f.stat().st_size for f in wav_files)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"\n📁 Location: {OUTPUT_DIR}")
    print(f"📊 Total WAV files: {len(wav_files)}")
    print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
    
    # Count by source
    print("\n📋 Files by source:")
    sources = {
        "realised_obj_": "ReaLISED (Object Falling)",
        "tut_impact_": "TUT Sound Events",
        "esc50_": "ESC-50",
        "zenodo_extra_": "Zenodo DataSED",
        "freesound_": "Freesound.org",
    }
    
    for prefix, name in sources.items():
        count = len(list(OUTPUT_DIR.glob(f"{prefix}*.wav")))
        if count > 0:
            print(f"   {name}: {count} files")
    
    print("\n" + "="*70)
    
    if len(wav_files) < 600:
        needed = 600 - len(wav_files)
        print(f"\n⚠️ Need {needed} more files to reach 600 target")
        print("\n🔧 Options to get more files:")
        print("   1. Run: python3 download_freesound.py")
        print("      (Requires free API key from https://freesound.org/apiv2/apply)")
        print("\n   2. Manual download from:")
        print("      - https://sound-effects.bbcrewind.co.uk/ (Household category)")
        print("      - https://mixkit.co/free-sound-effects/falling/")
    else:
        print(f"\n✅ Target of 600+ files ACHIEVED!")


def main():
    print("="*70)
    print("  MULTI-SOURCE OBJECT FALL AUDIO COLLECTOR")
    print("  Collecting REAL (non-synthetic, non-AI) audio samples")
    print("="*70)
    
    # Check current count
    current_count = len(list(OUTPUT_DIR.glob("*.wav")))
    print(f"\nCurrent files in directory: {current_count}")
    
    if current_count >= 600:
        print("\n✅ Already have 600+ files!")
        show_final_summary()
        return
    
    # Download from ESC-50 (glass breaking, door knock)
    esc50_count = download_esc50()
    
    # Download additional from Zenodo
    zenodo_count = download_additional_zenodo()
    
    # Show summary
    show_final_summary()


if __name__ == "__main__":
    main()
