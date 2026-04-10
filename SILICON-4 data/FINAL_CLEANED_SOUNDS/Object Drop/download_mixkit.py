#!/usr/bin/env python3
"""
Download free sound effects from Mixkit.co

Mixkit offers royalty-free, no-attribution sounds that can be downloaded
without an account. These are REAL recorded sounds.
"""

import requests
from pathlib import Path
import subprocess
import time
from bs4 import BeautifulSoup
import re

OUTPUT_DIR = Path(__file__).parent


def get_mixkit_sounds(category_url):
    """Scrape Mixkit for sound effect download URLs."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(category_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Find download links
        # Mixkit uses direct download links in their page
        # The format is usually https://assets.mixkit.co/sfx/download/mixkit-X.wav
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all sound card elements
        sounds = []
        
        # Look for download buttons/links
        download_links = soup.find_all('a', href=re.compile(r'mixkit.*\.(wav|mp3)'))
        
        for link in download_links:
            href = link.get('href')
            if href:
                sounds.append(href)
        
        return list(set(sounds))
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_file(url, output_path):
    """Download a file."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        return False


def convert_to_standard_format(input_path, output_path):
    """Convert to standard 7-second 44.1kHz mono WAV."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-t", "7",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except:
        return False


def main():
    print("="*70)
    print("  MIXKIT SOUND EFFECTS DOWNLOADER")
    print("="*70)
    
    # Mixkit URLs for falling/dropping/impact sounds
    categories = [
        "https://mixkit.co/free-sound-effects/falling/",
        "https://mixkit.co/free-sound-effects/drop/",
        "https://mixkit.co/free-sound-effects/crash/",
        "https://mixkit.co/free-sound-effects/impact/",
        "https://mixkit.co/free-sound-effects/glass/",
        "https://mixkit.co/free-sound-effects/breaking/",
    ]
    
    # We need to use known working URLs from Mixkit
    # Based on research, Mixkit uses asset IDs
    
    # Sample IDs for falling/impact sounds (discovered from their catalog)
    known_sound_ids = list(range(1, 2000))  # Try a range of IDs
    
    print("\nNote: Mixkit requires JavaScript to load sound previews.")
    print("Attempting to download using known asset patterns...")
    
    temp_dir = OUTPUT_DIR / "mixkit_temp"
    temp_dir.mkdir(exist_ok=True)
    
    downloaded = 0
    target = 50  # Try to get 50 sounds
    
    # Try different ID patterns
    base_urls = [
        "https://assets.mixkit.co/sfx/download/mixkit-{}.wav",
        "https://assets.mixkit.co/active_storage/sfx/download/mixkit-{}.wav",
    ]
    
    for sound_id in range(1, 500):  # Try first 500 IDs
        if downloaded >= target:
            break
        
        for base_url in base_urls:
            url = base_url.format(sound_id)
            temp_file = temp_dir / f"mixkit_{sound_id}.wav"
            
            if download_file(url, temp_file):
                # Convert to standard format
                output_path = OUTPUT_DIR / f"mixkit_{downloaded+1:04d}.wav"
                if convert_to_standard_format(temp_file, output_path):
                    downloaded += 1
                    print(f"  Downloaded sound {downloaded}")
                    temp_file.unlink(missing_ok=True)
                    break
                else:
                    temp_file.unlink(missing_ok=True)
        
        time.sleep(0.2)  # Rate limiting
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✅ Downloaded {downloaded} sounds from Mixkit")


if __name__ == "__main__":
    main()
