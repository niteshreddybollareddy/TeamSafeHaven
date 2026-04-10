#!/usr/bin/env python3
"""
Download cat meow audio files from Wikimedia Commons using their API.
"""

import requests
import os
import time

OUTPUT_DIR = "wikimedia_cat_sounds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wikimedia Commons API endpoint
API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "CatSoundDownloader/1.0 (educational research project)"
}

def get_category_files(category):
    """Get list of files from a Wikimedia Commons category."""
    files = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": "50",
        "format": "json"
    }
    
    try:
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching category {category}: {e}")
        return files
    
    if "query" in data and "categorymembers" in data["query"]:
        for member in data["query"]["categorymembers"]:
            files.append(member["title"])
    
    return files

def get_file_url(filename):
    """Get the direct download URL for a file."""
    params = {
        "action": "query",
        "titles": filename,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    data = response.json()
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if "imageinfo" in page_data:
            return page_data["imageinfo"][0]["url"]
    return None

def download_file(url, output_path):
    """Download a file from URL."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    print("Downloading cat sounds from Wikimedia Commons...")
    
    # Get files from cat meowing category
    files = get_category_files("Audio_files_of_cats_meowing")
    print(f"Found {len(files)} files in 'Audio files of cats meowing' category")
    
    # Also get purring sounds
    purr_files = get_category_files("Audio_files_of_cats_purring")
    print(f"Found {len(purr_files)} files in 'Audio files of cats purring' category")
    
    all_files = files + purr_files
    
    downloaded = 0
    for filename in all_files:
        # Get file extension
        ext = filename.split(".")[-1].lower()
        if ext not in ["wav", "ogg", "flac", "mp3"]:
            continue
        
        url = get_file_url(filename)
        if not url:
            print(f"Could not get URL for {filename}")
            continue
        
        # Create safe filename
        safe_name = filename.replace("File:", "").replace(" ", "_")
        output_path = os.path.join(OUTPUT_DIR, safe_name)
        
        if os.path.exists(output_path):
            print(f"Skipping {safe_name} (already exists)")
            downloaded += 1
            continue
        
        print(f"Downloading: {safe_name}")
        if download_file(url, output_path):
            downloaded += 1
            time.sleep(0.5)  # Be nice to the server
    
    print(f"\nDownloaded {downloaded} files to {OUTPUT_DIR}/")
    print(f"Total files in folder: {len(os.listdir(OUTPUT_DIR))}")

if __name__ == "__main__":
    main()
