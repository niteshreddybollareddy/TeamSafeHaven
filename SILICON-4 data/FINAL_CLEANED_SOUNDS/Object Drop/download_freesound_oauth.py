#!/usr/bin/env python3
"""
Freesound.org Object Falling/Dropping Sounds Downloader
Using official freesound-python library with OAuth2 authentication.

For OAuth2, we need to:
1. Get authorization URL
2. User visits URL and authorizes
3. Get redirect with code
4. Exchange code for access token
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import freesound

# ==================== CONFIGURATION ====================
CLIENT_ID = "2ibwRkflEmCW3DWA3kZg"
CLIENT_SECRET = "1JUMKw9QOcwSyk7ji81gAq4xsr6X3iQJJOTsyT5D"

TARGET_COUNT = 500
TARGET_DURATION = 7
OUTPUT_DIR = Path(__file__).parent
# ========================================================

# Specific search queries for object falling sounds
SEARCH_QUERIES = [
    "object falling", "object drop", "falling object", "dropping",
    "plate falling", "cup falling", "glass falling", "bottle falling",
    "book drop", "phone drop", "keys falling", "coin drop",
    "spoon falling", "fork falling", "pan falling", "pot falling",
    "bowl falling", "mug falling", "jar falling", "can falling",
    "box falling", "metal falling", "plastic falling", "wood falling",
    "ceramic falling", "impact floor", "drop thud", "object clatter",
    "kitchen object fall", "cutlery drop", "dish drop", "utensil falling",
    "household object drop", "domestic fall", "thing falling", "item drop",
]


def is_relevant_sound(name, description, tags):
    """Filter for actual object falling/dropping sounds."""
    text = f"{name} {description} {' '.join(tags or [])}".lower()
    
    # Exclude these
    exclude_terms = [
        'music', 'song', 'melody', 'rain', 'waterfall', 'water fall',
        'footstep', 'walk', 'voice', 'speech', 'talk', 'sing',
        'ambient', 'atmosphere', 'loop', 'drone', 'synth',
        'free fall', 'freefall', 'wind', 'thunder', 'nature',
        'drum', 'beat', 'guitar', 'piano', 'bass',
    ]
    
    for term in exclude_terms:
        if term in text:
            return False
    
    # Include these
    include_terms = [
        'fall', 'drop', 'falling', 'dropping',
        'impact', 'thud', 'clatter', 'crash',
        'object', 'item', 'thing',
    ]
    
    for term in include_terms:
        if term in text:
            return True
    
    return False


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
    print("  Using OAuth2 Authentication")
    print("="*70)
    
    # Initialize client
    client = freesound.FreesoundClient()
    client.set_token(CLIENT_ID, auth_type="client")
    
    print("\n🔐 OAuth2 Authorization Required")
    print("-"*70)
    
    # Get authorization URL
    auth_url = f"https://freesound.org/apiv2/oauth2/authorize/?client_id={CLIENT_ID}&response_type=code"
    
    print(f"\n1. Open this URL in your browser:\n")
    print(f"   {auth_url}")
    print(f"\n2. Login and click 'Authorize'")
    print(f"\n3. You'll be redirected to your redirect URL with a code.")
    print(f"   The URL will look like: http://your-redirect-url/?code=XXXXX")
    print(f"\n4. Copy the 'code' value from the URL and paste it below:")
    
    auth_code = input("\n   Authorization code: ").strip()
    
    if not auth_code:
        print("No code provided. Exiting.")
        return
    
    # Exchange code for access token
    print("\n🔄 Exchanging code for access token...")
    
    try:
        import requests
        token_url = "https://freesound.org/apiv2/oauth2/access_token/"
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": auth_code,
        }
        
        response = requests.post(token_url, data=data)
        
        if response.status_code != 200:
            print(f"Error getting token: {response.text}")
            return
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            print(f"No access token in response: {token_data}")
            return
        
        print("✅ Got access token!")
        
        # Set the access token
        client.set_token(access_token, auth_type="oauth")
        
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create temp directory
    temp_dir = OUTPUT_DIR / "freesound_temp"
    temp_dir.mkdir(exist_ok=True)
    
    downloaded = 0
    seen_ids = set()
    
    # Check existing
    for f in OUTPUT_DIR.glob("freesound_*.wav"):
        try:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                seen_ids.add(int(parts[1]))
        except:
            pass
    
    print(f"\nSkipping {len(seen_ids)} already downloaded")
    print(f"\n🔍 Searching for object falling sounds...")
    
    for query in SEARCH_QUERIES:
        if downloaded >= TARGET_COUNT:
            break
        
        print(f"\n🔍 Searching: '{query}'")
        
        try:
            results = client.text_search(
                query=query,
                filter="duration:[0.5 TO 20]",
                fields="id,name,description,tags,previews,duration"
            )
            
            for sound in results:
                if downloaded >= TARGET_COUNT:
                    break
                
                if sound.id in seen_ids:
                    continue
                
                # Check relevance
                tags = sound.tags if hasattr(sound, 'tags') else []
                desc = sound.description if hasattr(sound, 'description') else ""
                
                if not is_relevant_sound(sound.name, desc, tags):
                    continue
                
                seen_ids.add(sound.id)
                
                # Get preview URL
                try:
                    preview_url = sound.previews.preview_hq_mp3
                except:
                    try:
                        preview_url = sound.previews.preview_lq_mp3
                    except:
                        continue
                
                # Download preview
                temp_file = temp_dir / f"fs_{sound.id}.mp3"
                
                try:
                    import requests
                    resp = requests.get(preview_url, timeout=60)
                    resp.raise_for_status()
                    
                    with open(temp_file, 'wb') as f:
                        f.write(resp.content)
                    
                    # Convert to WAV
                    wav_path = OUTPUT_DIR / f"freesound_{sound.id}.wav"
                    
                    if convert_to_wav(temp_file, wav_path, TARGET_DURATION):
                        downloaded += 1
                        dur = sound.duration if hasattr(sound, 'duration') else 0
                        print(f"  ✅ [{downloaded}/{TARGET_COUNT}] {sound.name[:40]}... ({dur:.1f}s)")
                    
                except Exception as e:
                    temp_file.unlink(missing_ok=True)
                    continue
                
                time.sleep(0.3)
                
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            continue
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    total = len(list(OUTPUT_DIR.glob("*.wav")))
    
    print("\n" + "="*70)
    print(f"✅ Downloaded {downloaded} files from Freesound")
    print(f"📊 Total WAV files: {total}")
    print("="*70)


if __name__ == "__main__":
    main()
