#!/usr/bin/env python3
"""
Freesound.org Domestic Object Falling Sounds Downloader
- NO GLASS sounds
- Only daily domestic/household objects
- Real recordings only, no AI generated
"""

import os
import time
import requests
from pathlib import Path
import subprocess

# ==================== CONFIGURATION ====================
API_KEY = "1JUMKw9QOcwSyk7ji81gAq4xsr6X3iQJJOTsyT5D"

TARGET_COUNT = 150  # Need ~135 more to reach 500
TARGET_DURATION = 7
OUTPUT_DIR = Path(__file__).parent
# ========================================================

# NEW search queries for more domestic objects (NO GLASS)
SEARCH_QUERIES = [
    # More specific metal sounds
    "tin dropping",
    "aluminum falling",
    "steel object drop",
    "iron falling",
    "metallic thud",
    "metallic clatter",
    "saucepan drop",
    "baking tray drop",
    "cookie sheet fall",
    "tea kettle drop",
    "colander falling",
    "strainer drop",
    "grater falling",
    "tongs drop",
    "serving spoon drop",
    "butter knife drop",
    "steak knife fall",
    "measuring spoon",
    "measuring cup metal",
    
    # More plastic items
    "tupperware falling",
    "food container drop",
    "storage box drop",
    "lid popping",
    "bottle cap drop",
    "pen cap drop",
    "flashlight drop",
    "hair brush drop",
    "comb falling",
    "toothbrush falling",
    "soap dispenser",
    "hand sanitizer drop",
    "water bottle drop",
    "thermos drop",
    "lunch box drop",
    "plastic bin drop",
    "trash can drop",
    "recycling bin",
    "laundry basket",
    "hanger drop",
    "clothes hanger",
    
    # More wooden items
    "chair falling",
    "stool falling",
    "table knock",
    "drawer slam",
    "cabinet door",
    "door knock",
    "door slam",
    "picture frame drop",
    "photo frame fall",
    "clock falling",
    "wall clock drop",
    "wooden toy",
    "building blocks",
    "toy blocks drop",
    "domino falling",
    "dice drop",
    "chess piece drop",
    "ruler drop",
    "yardstick fall",
    "broom falling",
    "mop drop",
    "dustpan drop",
    
    # More ceramic/porcelain
    "teacup drop",
    "coffee cup drop",
    "saucer drop",
    "teapot drop",
    "vase falling",
    "flower pot drop",
    "plant pot fall",
    "planter drop",
    "decorative plate",
    "serving dish",
    "casserole dish",
    "baking dish drop",
    "pie dish fall",
    "ramekin drop",
    "soup bowl drop",
    "cereal bowl",
    "rice bowl drop",
    "salad bowl",
    
    # Electronics specific
    "remote control thud",
    "tv remote drop",
    "game controller drop",
    "joystick drop",
    "headphones drop",
    "earbuds drop",
    "speaker drop",
    "bluetooth speaker",
    "power bank drop",
    "usb cable drop",
    "charger cable",
    "extension cord",
    "plug drop",
    "adapter drop",
    "calculator drop",
    "alarm clock drop",
    "digital clock",
    "watch drop",
    "wristwatch fall",
    
    # Toys and games
    "toy car drop",
    "action figure drop",
    "doll falling",
    "stuffed animal",
    "teddy bear drop",
    "lego drop",
    "building block",
    "puzzle piece",
    "board game piece",
    "marble drop",
    "ball drop floor",
    "bouncy ball",
    "tennis ball drop",
    "basketball drop",
    "rubber ball",
    "toy truck",
    "toy train",
    
    # More kitchen sounds
    "measuring jug",
    "mixing bowl drop",
    "blender jar drop",
    "coffee mug drop",
    "tea mug drop",
    "thermos flask",
    "ice cube tray",
    "egg timer drop",
    "salt shaker",
    "pepper mill drop",
    "spice jar drop",
    "condiment bottle",
    "ketchup bottle",
    "mayonnaise jar",
    "jam jar drop",
    "honey jar",
    "olive oil bottle",
    "cooking oil",
    "vinegar bottle",
    
    # Office items
    "paperclip drop",
    "tape dispenser",
    "hole punch drop",
    "binder drop",
    "file folder",
    "document drop",
    "envelope drop",
    "stamp drop",
    "ink pad",
    "rubber stamp",
    "whiteboard marker",
    "marker drop",
    "highlighter drop",
    "crayon drop",
    "colored pencil",
    "paintbrush drop",
    "art supplies",
    
    # Miscellaneous household
    "umbrella falling",
    "umbrella drop",
    "walking stick",
    "cane drop",
    "flashlight fall",
    "torch drop",
    "lighter drop",
    "matches box",
    "candle drop",
    "candle holder",
    "picture falling",
    "mirror falling",
    "frame drop",
    "ornament falling",
    "decoration drop",
    "christmas ornament",
    "trophy drop",
    "medal drop",
    "badge drop",
    "keychain drop",
    "keyring drop",
    "wallet drop",
    "purse drop",
    "sunglasses drop",
    "spectacles drop",
    
    # Foley sounds (often labeled this way)
    "foley drop",
    "foley falling",
    "foley thud",
    "foley impact",
    "foley object",
    "prop drop",
    "prop falling",
    "sound effect drop",
    "sfx drop",
    "sfx falling",
]


def search_freesound(query, page=1, page_size=50):
    """Search Freesound using API key."""
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
        
        if response.status_code != 200:
            return {"results": []}
        
        return response.json()
    except Exception as e:
        return {"results": []}


def is_valid_sound(name, description, tags):
    """
    Filter for domestic object falling sounds.
    EXCLUDE glass sounds.
    """
    text = f"{name} {description} {' '.join(tags or [])}".lower()
    
    # ====== STRICTLY EXCLUDE ======
    exclude_terms = [
        # Glass - user doesn't want glass
        'glass', 'glasses', 'glassy', 'window', 'windshield', 'mirror',
        'bottle break', 'bottle smash', 'bottle shatter',
        
        # Music/ambient - not object sounds
        'music', 'song', 'melody', 'musical', 'tune', 'soundtrack',
        'ambient', 'ambience', 'atmosphere', 'background noise',
        'loop', 'looping', 'drone', 'synth', 'synthesizer',
        'drum', 'drums', 'beat', 'rhythm', 'percussion',
        'guitar', 'piano', 'bass', 'violin', 'flute', 'trumpet',
        
        # Nature/weather - not domestic
        'rain', 'waterfall', 'water fall', 'rainfall', 'thunder',
        'wind', 'storm', 'nature', 'forest', 'ocean', 'river', 'stream',
        
        # Human sounds
        'voice', 'speech', 'talk', 'talking', 'speaking', 'sing', 'singing',
        'scream', 'shout', 'whisper', 'laugh', 'cry',
        'footstep', 'footsteps', 'walk', 'walking', 'running', 'steps',
        
        # Wrong context
        'free fall', 'freefall', 'skydive', 'parachute',
        'game', 'video game', '8bit', '8-bit', 'arcade',
        'sci-fi', 'scifi', 'laser', 'space', 'alien',
        'horror', 'scary', 'monster', 'ghost', 'zombie',
        'explosion', 'explode', 'bomb', 'war', 'gun', 'shot', 'bullet',
        'car', 'vehicle', 'engine', 'motor', 'traffic',
        'animal', 'bird', 'dog', 'cat', 'bark', 'meow',
        'construction', 'machinery', 'factory',
        'bell', 'alarm', 'siren', 'notification',
    ]
    
    for term in exclude_terms:
        if term in text:
            return False
    
    # ====== MUST INCLUDE at least one ======
    include_terms = [
        # Actions
        'fall', 'falling', 'fell', 'drop', 'dropping', 'dropped',
        'impact', 'hit', 'hitting', 'thud', 'thump', 'bump',
        'clatter', 'clang', 'clink', 'crash', 'knock',
        
        # Objects
        'object', 'item', 'thing',
        'metal', 'plastic', 'wood', 'wooden', 'ceramic', 'porcelain',
        'plate', 'cup', 'bowl', 'mug', 'dish', 'pot', 'pan', 'lid',
        'spoon', 'fork', 'knife', 'cutlery', 'utensil',
        'book', 'phone', 'keys', 'key', 'coin', 'coins', 'can',
        'box', 'container', 'bottle', 'jar', 'bucket',
        'remote', 'toy', 'tool', 'hammer', 'wrench',
        'pencil', 'pen', 'shoe', 'bag',
        
        # Context
        'floor', 'ground', 'surface', 'table', 'desk',
        'kitchen', 'household', 'domestic', 'home',
    ]
    
    for term in include_terms:
        if term in text:
            return True
    
    return False


def download_preview(sound_id, preview_url, temp_dir):
    """Download preview MP3."""
    try:
        response = requests.get(preview_url, timeout=60)
        response.raise_for_status()
        
        filepath = temp_dir / f"fs_{sound_id}.mp3"
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filepath
    except:
        return None


def convert_to_wav(input_path, output_path, target_duration=7):
    """Convert to WAV."""
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
    print("  FREESOUND - Domestic Object Falling Sounds (NO GLASS)")
    print(f"  Target: {TARGET_COUNT} files")
    print("="*70)
    
    # Test API
    print("\n🔄 Testing API...")
    test = search_freesound("test", page=1, page_size=1)
    if "results" not in test and "count" not in test:
        print("❌ API failed!")
        return
    print("✅ API OK!")
    
    temp_dir = OUTPUT_DIR / "temp_downloads"
    temp_dir.mkdir(exist_ok=True)
    
    downloaded = 0
    seen_ids = set()
    
    # Load existing IDs
    for f in OUTPUT_DIR.glob("*.wav"):
        try:
            # Extract ID from various naming patterns
            stem = f.stem
            if "freesound_" in stem:
                seen_ids.add(stem.split("freesound_")[1].split("_")[0])
            elif stem.isdigit():
                seen_ids.add(stem)
        except:
            pass
    
    print(f"\nSkipping {len(seen_ids)} existing sounds")
    print(f"\n🔍 Searching for domestic objects (NO GLASS)...\n")
    
    for query in SEARCH_QUERIES:
        if downloaded >= TARGET_COUNT:
            break
        
        print(f"🔍 '{query}'", end="", flush=True)
        count_this_query = 0
        
        for page in range(1, 4):
            if downloaded >= TARGET_COUNT:
                break
            
            results = search_freesound(query, page=page, page_size=50)
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
                desc = sound.get("description", "")
                tags = sound.get("tags", [])
                
                if not is_valid_sound(name, desc, tags):
                    continue
                
                seen_ids.add(sound_id)
                
                previews = sound.get("previews", {})
                preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                
                if not preview_url:
                    continue
                
                mp3_path = download_preview(sound_id, preview_url, temp_dir)
                
                if mp3_path and mp3_path.exists():
                    wav_path = OUTPUT_DIR / f"freesound_{sound_id}.wav"
                    
                    if convert_to_wav(mp3_path, wav_path, TARGET_DURATION):
                        downloaded += 1
                        count_this_query += 1
                
                time.sleep(0.2)
            
            time.sleep(0.3)
        
        if count_this_query > 0:
            print(f" - {count_this_query} files ({downloaded} total)")
        else:
            print(" - 0")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    total = len(list(OUTPUT_DIR.glob("*.wav")))
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Downloaded: {downloaded} new files")
    print(f"Total WAV files: {total}")
    print(f"Location: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
