#!/usr/bin/env python3
"""
Freesound.org Object Falling/Dropping Sounds Downloader
Using the API key (client secret) for token-based authentication.
"""

import os
import time
import requests
from pathlib import Path
import subprocess

# ==================== CONFIGURATION ====================
# From Freesound API credentials page:
# The "Client secret/Api key" can be used as token for API access
API_KEY = "1JUMKw9QOcwSyk7ji81gAq4xsr6X3iQJJOTsyT5D"

TARGET_COUNT = 500
TARGET_DURATION = 7
OUTPUT_DIR = Path(__file__).parent
# ========================================================

# Specific search queries for object falling sounds
SEARCH_QUERIES = [
    # Primary object falling terms
    "object falling",
    "object drop", 
    "falling object",
    "dropping object",
    "thing falling",
    "item falling",
    "item drop",
    
    # Specific domestic objects
    "plate falling",
    "plate drop",
    "cup falling", 
    "cup drop",
    "glass falling",
    "glass drop",
    "glass breaking fall",
    "bottle falling",
    "bottle drop",
    "book falling",
    "book drop",
    "phone falling",
    "phone drop",
    "keys falling",
    "keys drop",
    "coin falling",
    "coin drop",
    "coins falling",
    "spoon falling",
    "spoon drop",
    "fork falling",
    "fork drop",
    "knife falling",
    "pan falling",
    "pan drop",
    "pot falling",
    "pot lid",
    "bowl falling",
    "bowl drop",
    "mug falling",
    "mug drop",
    "jar falling",
    "jar drop",
    "can falling",
    "can drop",
    "box falling",
    "box drop",
    "remote falling",
    "toy falling",
    "toy drop",
    "ball falling",
    "ball drop",
    "ball bounce",
    "pencil falling",
    "pencil drop",
    "pen falling",
    "pen drop",
    
    # Materials
    "metal falling",
    "metal drop",
    "metal object",
    "plastic falling",
    "plastic drop",
    "wood falling",
    "wooden drop",
    "ceramic falling",
    "ceramic drop",
    "ceramic break",
    
    # Impact sounds
    "object impact",
    "drop impact",
    "fall impact",
    "thud floor",
    "object thud",
    "thud sound",
    "clatter floor",
    "object clatter",
    "clatter sound",
    
    # Kitchen specific
    "kitchen object",
    "cutlery drop",
    "cutlery falling",
    "dish drop",
    "dish falling",
    "crockery drop",
    "utensil falling",
    "utensil drop",
    
    # Household
    "household object",
    "domestic object",
    "home object drop",
]


def search_freesound(query, page=1, page_size=50):
    """Search Freesound using API key token."""
    url = "https://freesound.org/apiv2/search/text/"
    
    params = {
        "query": query,
        "page": page,
        "page_size": page_size,
        "fields": "id,name,previews,duration,description,tags",
        "filter": "duration:[0.5 TO 20]",
        "token": API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 401:
            print(f"  ⚠️ Auth error (401)")
            return {"results": [], "error": "auth"}
        
        if response.status_code != 200:
            print(f"  ⚠️ HTTP {response.status_code}: {response.text[:100]}")
            return {"results": []}
        
        return response.json()
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        return {"results": []}


def is_relevant_sound(name, description, tags):
    """Filter for actual object falling/dropping sounds."""
    text = f"{name} {description} {' '.join(tags or [])}".lower()
    
    # Definitely exclude these (wrong type of sounds)
    exclude_terms = [
        'music', 'song', 'melody', 'musical', 'tune',
        'rain', 'waterfall', 'water fall', 'rainfall',
        'footstep', 'footsteps', 'walk', 'walking', 'running',
        'voice', 'speech', 'talk', 'talking', 'speaking', 'sing', 'singing',
        'ambient', 'ambience', 'atmosphere', 'background',
        'loop', 'looping', 'drone', 'synth', 'synthesizer',
        'free fall', 'freefall', 'skydive',
        'wind', 'thunder', 'storm', 'nature', 'forest', 'ocean',
        'drum', 'drums', 'beat', 'rhythm',
        'guitar', 'piano', 'bass', 'violin', 'flute',
        'game', 'video game', '8bit', '8-bit',
        'sci-fi', 'scifi', 'laser', 'space',
        'horror', 'scary', 'monster',
        'explosion', 'explode', 'bomb', 'war', 'gun', 'shot',
        'car', 'vehicle', 'engine', 'motor',
        'animal', 'bird', 'dog', 'cat',
    ]
    
    for term in exclude_terms:
        if term in text:
            return False
    
    # Must have at least one of these (relevant sounds)
    include_terms = [
        'fall', 'falling', 'fell',
        'drop', 'dropping', 'dropped',
        'impact', 'hit', 'hitting',
        'thud', 'thump', 'bump',
        'clatter', 'clang', 'clink',
        'crash', 'smash', 'break', 'broken',
        'object', 'item', 'thing',
        'floor', 'ground', 'surface', 'table',
        'kitchen', 'household', 'domestic',
        'metal', 'plastic', 'wood', 'glass', 'ceramic',
        'plate', 'cup', 'bowl', 'mug', 'glass', 'bottle',
        'pan', 'pot', 'lid', 'spoon', 'fork', 'knife',
        'book', 'phone', 'keys', 'coin', 'box',
    ]
    
    for term in include_terms:
        if term in text:
            return True
    
    return False


def download_preview(sound_id, preview_url, temp_dir):
    """Download the HQ preview MP3."""
    try:
        response = requests.get(preview_url, timeout=60)
        response.raise_for_status()
        
        filepath = temp_dir / f"fs_{sound_id}.mp3"
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filepath
    except Exception as e:
        return None


def convert_to_wav(input_path, output_path, target_duration=7):
    """Convert to standard WAV format."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-t", str(target_duration),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        input_path.unlink(missing_ok=True)
        
        return result.returncode == 0 and output_path.exists()
    except:
        input_path.unlink(missing_ok=True)
        return False


def main():
    print("="*70)
    print("  FREESOUND.ORG - Object Falling Sounds Downloader")
    print(f"  Target: {TARGET_COUNT} files, {TARGET_DURATION}s each")
    print("="*70)
    
    # Test API connection first
    print("\n🔄 Testing API connection...")
    test_result = search_freesound("test", page=1, page_size=1)
    
    if test_result.get("error") == "auth":
        print("\n❌ Authentication failed!")
        print("The API key might not be valid or might need OAuth2.")
        return
    
    if "results" not in test_result and "count" not in test_result:
        print("\n❌ API connection failed!")
        return
    
    print("✅ API connection successful!")
    
    # Create temp directory
    temp_dir = OUTPUT_DIR / "freesound_temp"
    temp_dir.mkdir(exist_ok=True)
    
    downloaded = 0
    seen_ids = set()
    
    # Check existing files
    for f in OUTPUT_DIR.glob("freesound_*.wav"):
        try:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                seen_ids.add(parts[1])
        except:
            pass
    
    print(f"\nSkipping {len(seen_ids)} already downloaded sounds")
    print(f"\n🔍 Searching for object falling sounds...\n")
    
    for query in SEARCH_QUERIES:
        if downloaded >= TARGET_COUNT:
            print(f"\n✅ Reached target of {TARGET_COUNT} files!")
            break
        
        print(f"🔍 '{query}'", end="", flush=True)
        
        sounds_from_query = 0
        
        for page in range(1, 4):  # Up to 3 pages per query
            if downloaded >= TARGET_COUNT:
                break
            
            results = search_freesound(query, page=page, page_size=50)
            
            if results.get("error") == "auth":
                print(" - auth error")
                break
            
            sounds = results.get("results", [])
            if not sounds:
                break
            
            for sound in sounds:
                if downloaded >= TARGET_COUNT:
                    break
                
                sound_id = str(sound["id"])
                if sound_id in seen_ids:
                    continue
                
                name = sound.get("name", "")
                description = sound.get("description", "")
                tags = sound.get("tags", [])
                duration = sound.get("duration", 0)
                
                # Filter for relevant sounds
                if not is_relevant_sound(name, description, tags):
                    continue
                
                seen_ids.add(sound_id)
                
                # Get preview URL
                previews = sound.get("previews", {})
                preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                
                if not preview_url:
                    continue
                
                # Download
                mp3_path = download_preview(sound_id, preview_url, temp_dir)
                
                if mp3_path and mp3_path.exists():
                    wav_path = OUTPUT_DIR / f"freesound_{sound_id}.wav"
                    
                    if convert_to_wav(mp3_path, wav_path, TARGET_DURATION):
                        downloaded += 1
                        sounds_from_query += 1
                
                time.sleep(0.2)
            
            time.sleep(0.3)
        
        if sounds_from_query > 0:
            print(f" - got {sounds_from_query} files ({downloaded} total)")
        else:
            print(" - 0 relevant")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    total_wav = len(list(OUTPUT_DIR.glob("*.wav")))
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Downloaded from Freesound: {downloaded} files")
    print(f"Total WAV files in directory: {total_wav}")
    print(f"Location: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
