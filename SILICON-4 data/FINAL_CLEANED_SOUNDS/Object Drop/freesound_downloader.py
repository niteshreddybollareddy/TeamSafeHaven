#!/usr/bin/env python3
"""
Freesound.org Bulk Downloader for Object Falling Sounds

Requirements:
    pip install requests tqdm pydub

Setup:
    1. Create account at https://freesound.org/
    2. Get API key at https://freesound.org/apiv2/apply  
    3. Set your API key below or as environment variable FREESOUND_API_KEY
"""

import os
import time
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess

# ==================== CONFIGURATION ====================
API_KEY = os.environ.get("FREESOUND_API_KEY", "YOUR_API_KEY_HERE")
TARGET_COUNT = 600
TARGET_DURATION = 7  # seconds
OUTPUT_DIR = Path(__file__).parent
# ========================================================

SEARCH_QUERIES = [
    "object falling", "object drop", "object impact", "dropping object",
    "glass breaking", "metal falling", "plate drop", "cup falling",
    "book drop", "bottle falling", "keys dropping", "phone drop",
    "box falling", "can falling", "spoon drop", "fork falling",
    "pot lid drop", "ball bouncing", "plastic falling", "wooden object fall",
    "dish breaking", "vase falling", "remote falling", "toy drop",
    "kitchen object", "household drop", "domestic object", "item falling",
    "thud impact", "crash sound", "clatter", "thump", "bang",
    "ceramic breaking", "crockery drop", "cutlery drop", "pan drop",
    "jar falling", "container drop", "bucket falling", "tool drop",
]


def search_freesound(query, page=1, page_size=50):
    """Search Freesound for sounds matching query."""
    url = "https://freesound.org/apiv2/search/text/"
    params = {
        "query": query,
        "page": page,
        "page_size": page_size,
        "fields": "id,name,previews,duration,description,tags,download",
        "filter": "duration:[1 TO 20]",
        "token": API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error searching: {e}")
        return {"results": []}


def download_preview(sound_id, name, preview_url, output_dir):
    """Download the preview (high-quality MP3, no OAuth required)."""
    try:
        # Use HQ preview (128kbps MP3)
        response = requests.get(preview_url, timeout=60)
        response.raise_for_status()
        
        # Clean filename
        clean_name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
        clean_name = clean_name[:50]  # Limit length
        
        filename = f"{sound_id}_{clean_name}.mp3"
        filepath = output_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filepath
    except Exception as e:
        print(f"Error downloading {sound_id}: {e}")
        return None


def convert_to_wav_7s(input_path, output_dir, target_duration=7):
    """Convert audio to WAV format with target duration using ffmpeg."""
    output_path = output_dir / (input_path.stem + ".wav")
    
    try:
        # Use ffmpeg to convert and trim/pad to target duration
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-t", str(target_duration),  # Trim to max duration
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "44100",  # Sample rate
            "-ac", "1",  # Mono
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Remove original MP3
        input_path.unlink()
        
        return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("=" * 70)
        print("⚠️  ERROR: Please set your Freesound API key!")
        print("=" * 70)
        print("\n1. Create account at https://freesound.org/")
        print("2. Get API key at https://freesound.org/apiv2/apply")
        print("3. Set it in this script or as environment variable FREESOUND_API_KEY")
        print("\nExample:")
        print("  export FREESOUND_API_KEY='your-api-key-here'")
        print("  python freesound_downloader.py")
        return
    
    print("=" * 70)
    print("Freesound.org Object Falling Sounds Downloader")
    print(f"Target: {TARGET_COUNT} WAV files, {TARGET_DURATION} seconds each")
    print("=" * 70)
    
    output_dir = OUTPUT_DIR / "freesound_downloads"
    output_dir.mkdir(exist_ok=True)
    
    final_output = OUTPUT_DIR
    
    downloaded = 0
    seen_ids = set()
    
    for query in SEARCH_QUERIES:
        if downloaded >= TARGET_COUNT:
            print(f"\n✅ Reached target of {TARGET_COUNT} files!")
            break
        
        print(f"\n🔍 Searching: '{query}'")
        
        for page in range(1, 4):  # Get up to 3 pages per query
            if downloaded >= TARGET_COUNT:
                break
                
            results = search_freesound(query, page=page)
            
            if not results.get("results"):
                break
            
            for sound in results["results"]:
                if downloaded >= TARGET_COUNT:
                    break
                
                sound_id = sound["id"]
                if sound_id in seen_ids:
                    continue
                seen_ids.add(sound_id)
                
                name = sound.get("name", f"sound_{sound_id}")
                duration = sound.get("duration", 0)
                
                # Get HQ preview URL
                previews = sound.get("previews", {})
                preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                
                if not preview_url:
                    continue
                
                print(f"  📥 Downloading: {name[:40]}... ({duration:.1f}s)")
                
                # Download preview
                mp3_path = download_preview(sound_id, name, preview_url, output_dir)
                
                if mp3_path:
                    # Convert to WAV with target duration
                    wav_path = convert_to_wav_7s(mp3_path, final_output, TARGET_DURATION)
                    if wav_path:
                        downloaded += 1
                        print(f"      ✅ Saved: {wav_path.name} ({downloaded}/{TARGET_COUNT})")
                
                # Rate limiting
                time.sleep(0.5)
            
            time.sleep(1)  # Pause between pages
    
    # Cleanup temp directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    
    print(f"\n{'='*70}")
    print(f"✅ Downloaded {downloaded} WAV files to: {final_output}")
    print("="*70)


if __name__ == "__main__":
    main()
