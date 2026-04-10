#!/usr/bin/env python3
"""
Download more real audio datasets:
- TUT Sound Events 2016 (Home indoor sounds)
- UrbanSound8K (glass break, etc.)
- FSDKaggle2018 (focused subset)
"""

import os
import requests
import zipfile  
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil
import random

OUTPUT_DIR = Path(__file__).parent


def download_file(url, output_path, desc="Downloading"):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f:
            if total_size:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                print(f"  Downloading (unknown size)...")
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def download_tut_2016_home():
    """Download TUT Sound Events 2016 - Home indoor recordings."""
    print("\n" + "="*70)
    print("📦 TUT SOUND EVENTS 2016 - Home Indoor Sounds")
    print("="*70)
    
    # Direct link to TUT 2016 development audio
    url = "https://zenodo.org/record/45759/files/TUT-sound-events-2016-development.audio.zip?download=1"
    
    zip_path = OUTPUT_DIR / "tut_2016.zip"
    extract_dir = OUTPUT_DIR / "tut_temp"
    
    print("📥 Downloading TUT Sound Events 2016 (0.9GB)...")
    if not download_file(url, zip_path, "TUT 2016"):
        return 0
    
    print("\n📦 Extracting...")
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        print(f"Error extracting: {e}")
        zip_path.unlink(missing_ok=True)
        return 0
    
    # Find audio files in home recordings
    audio_files = list(extract_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    # TUT 2016 has long recordings - we'll extract 7-second segments
    processed = 0
    for audio_file in audio_files:
        try:
            # Get duration
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", 
                        "format=duration", "-of", "default=nw=1:nk=1", str(audio_file)]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            try:
                duration = float(result.stdout.strip())
            except:
                continue
            
            # Extract multiple 7-second segments from each recording
            num_segments = int(duration / 7)
            for seg in range(min(num_segments, 10)):  # Max 10 segments per file
                start = seg * 7
                output_name = f"tut_home_{processed+1:04d}.wav"
                output_path = OUTPUT_DIR / output_name
                
                cmd = [
                    "ffmpeg", "-y", 
                    "-i", str(audio_file),
                    "-ss", str(start),
                    "-t", "7",
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "1",
                    str(output_path)
                ]
                
                subprocess.run(cmd, capture_output=True, timeout=30)
                if output_path.exists() and output_path.stat().st_size > 1000:
                    processed += 1
                    
                if processed >= 200:  # Limit per source
                    break
            
            if processed >= 200:
                break
                
        except Exception as e:
            continue
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    print(f"✅ Extracted {processed} segments from TUT 2016")
    return processed


def download_urbansound8k_subset():
    """Download from UrbanSound8K (glass break class focused)."""
    print("\n" + "="*70)
    print("📦 URBANSOUND8K - Glass Break & Impacts")
    print("="*70)
    
    print("Note: UrbanSound8K requires registration at:")
    print("https://urbansounddataset.weebly.com/urbansound8k.html")
    print("\nSkipping automatic download...")
    return 0


def download_fsdkaggle2018():
    """Download FSDKaggle2018 dataset - smaller curated subset."""
    print("\n" + "="*70)
    print("📦 FSD KAGGLE 2018 - Audio Tagging Challenge Data")  
    print("="*70)
    
    # This is a smaller focused dataset
    url = "https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip?download=1"
    
    zip_path = OUTPUT_DIR / "fsdkaggle2018.zip"
    extract_dir = OUTPUT_DIR / "fsd_temp"
    
    print("📥 Downloading FSD Kaggle 2018 (4.5GB - This will take a while)...")
    print("(Freesound audio tagging dataset with 41 categories)")
    
    if not download_file(url, zip_path, "FSD Kaggle 2018"):
        return 0
    
    print("\n📦 Extracting...")
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        print(f"Error extracting: {e}")
        zip_path.unlink(missing_ok=True)
        return 0
    
    # Find audio files
    audio_files = list(extract_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    # Shuffle and take a random sample for variety
    random.shuffle(audio_files)
    
    processed = 0
    for audio_file in audio_files[:300]:  # Take up to 300 files
        try:
            output_name = f"fsd_kaggle_{processed+1:04d}.wav"
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
            if output_path.exists() and output_path.stat().st_size > 1000:
                processed += 1
                
        except Exception as e:
            continue
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    
    print(f"✅ Processed {processed} files from FSD Kaggle 2018")
    return processed


def show_summary():
    """Show current file count."""
    wav_files = list(OUTPUT_DIR.glob("*.wav"))
    total_size = sum(f.stat().st_size for f in wav_files)
    
    print("\n" + "="*70)
    print("CURRENT STATUS")
    print("="*70)
    print(f"📊 Total WAV files: {len(wav_files)}")
    print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
    
    if len(wav_files) >= 600:
        print("\n✅ TARGET OF 600+ FILES REACHED!")
    else:
        print(f"\n⚠️ Need {600 - len(wav_files)} more files")


def main():
    print("="*70)
    print("  DOWNLOADING MORE REAL AUDIO DATASETS")
    print("="*70)
    
    current = len(list(OUTPUT_DIR.glob("*.wav")))
    print(f"\nCurrent files: {current}")
    
    if current >= 600:
        print("✅ Already have 600+ files!")
        show_summary()
        return
    
    # Try TUT 2016
    download_tut_2016_home()
    
    current = len(list(OUTPUT_DIR.glob("*.wav")))
    if current >= 600:
        show_summary()
        return
    
    # If still need more, try FSD Kaggle (warning: large download)
    print(f"\nCurrent files: {current}")
    if current < 600:
        print("\n⚠️ Still need more files. Trying FSD Kaggle 2018...")
        print("(This is a larger download ~4.5GB)")
        download_fsdkaggle2018()
    
    show_summary()


if __name__ == "__main__":
    main()
