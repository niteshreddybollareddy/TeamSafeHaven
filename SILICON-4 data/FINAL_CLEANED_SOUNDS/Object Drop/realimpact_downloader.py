#!/usr/bin/env python3
"""
RealImpact Dataset Processor

This script clones the RealImpact dataset and extracts audio files
for domestic object falling/impact sounds.

The RealImpact dataset contains 150,000+ REAL recordings of everyday
household objects being impacted.
"""

import os
import subprocess
from pathlib import Path
import shutil
import random


OUTPUT_DIR = Path(__file__).parent
TARGET_COUNT = 600
TARGET_DURATION = 7


def clone_and_download():
    """Clone RealImpact repo and download dataset."""
    
    repo_dir = OUTPUT_DIR / "RealImpact_temp"
    
    print("=" * 70)
    print("RealImpact Dataset Downloader")
    print("=" * 70)
    
    # Clone repository
    print("\n📥 Cloning RealImpact repository...")
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", 
            "https://github.com/samuel-clarke/RealImpact.git",
            str(repo_dir)
        ], check=True)
    
    # Download dataset
    print("\n📥 Downloading dataset (this may take a while)...")
    dataset_dir = repo_dir / "dataset"
    os.chdir(dataset_dir)
    
    # Run the download script
    subprocess.run(["bash", "download.sh"], check=True)
    
    print("\n✅ Dataset downloaded!")
    return dataset_dir


def process_audio_files(dataset_dir, target_count=600, target_duration=7):
    """Process and copy audio files to output directory."""
    
    print(f"\n🔄 Processing audio files...")
    
    # Find all WAV files in the dataset
    wav_files = list(dataset_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in dataset")
    
    # Shuffle and select
    random.shuffle(wav_files)
    selected = wav_files[:target_count]
    
    processed = 0
    for i, wav_file in enumerate(selected):
        try:
            output_name = f"realimpact_{i+1:04d}.wav"
            output_path = OUTPUT_DIR / output_name
            
            # Convert to target duration using ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", str(wav_file),
                "-t", str(target_duration),
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            processed += 1
            
            if processed % 50 == 0:
                print(f"  Processed {processed}/{target_count}")
                
        except Exception as e:
            print(f"  Error processing {wav_file}: {e}")
            continue
    
    print(f"\n✅ Processed {processed} files to: {OUTPUT_DIR}")


def cleanup(repo_dir):
    """Remove temporary downloaded repository."""
    print("\n🧹 Cleaning up temporary files...")
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    print("✅ Cleanup complete")


def main():
    try:
        dataset_dir = clone_and_download()
        process_audio_files(dataset_dir, TARGET_COUNT, TARGET_DURATION)
        cleanup(OUTPUT_DIR / "RealImpact_temp")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease try manually:")
        print("1. git clone https://github.com/samuel-clarke/RealImpact.git")
        print("2. cd RealImpact/dataset && bash download.sh")


if __name__ == "__main__":
    main()
