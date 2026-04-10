#!/usr/bin/env python3
"""
Filter ReaLISED files to keep only object falling sounds,
and organize the dataset properly.

ReaLISED naming convention:
- Files starting with 'obj' = object falling
- Other prefixes represent other indoor sounds (door, water, speech, etc.)

We want ONLY object falling sounds.
"""

import os
import shutil
from pathlib import Path

# Configuration
OBJECT_DROP_DIR = Path(__file__).parent
KEEP_ONLY_OBJECT_FALLING = True

# ReaLISED category prefixes (based on the dataset documentation)
OBJECT_FALLING_PREFIXES = [
    'obj',  # Object falling
]

# Additional relevant categories we might want (impact-related)
IMPACT_RELATED_PREFIXES = [
    'fur',  # Furniture movement (may have impacts)
    'dor',  # Door (closing sounds)
    'drw',  # Drawer
    'cup',  # Cupboard/wardrobe
]


def count_files_by_category():
    """Count and categorize all the downloaded files."""
    
    wav_files = list(OBJECT_DROP_DIR.glob("realised_*.wav"))
    
    print(f"\n📊 Total files: {len(wav_files)}")
    print("\nNote: Files were renamed during download, losing original category info.")
    print("The ReaLISED dataset structure is embedded in original FLAC filenames.")
    
    return wav_files


def keep_random_sample(count=600):
    """Keep a random sample of files for the dataset."""
    import random
    
    wav_files = list(OBJECT_DROP_DIR.glob("realised_*.wav"))
    
    print(f"\n📊 Total files available: {len(wav_files)}")
    
    if len(wav_files) <= count:
        print(f"Keeping all {len(wav_files)} files")
        return wav_files
    
    # Shuffle and keep desired count
    random.shuffle(wav_files)
    keep_files = set(f.name for f in wav_files[:count])
    remove_files = wav_files[count:]
    
    print(f"\n🗑️ Removing {len(remove_files)} excess files to reach target of {count}...")
    
    for f in remove_files:
        f.unlink()
    
    remaining = list(OBJECT_DROP_DIR.glob("realised_*.wav"))
    print(f"✅ Kept {len(remaining)} files")
    
    return list(OBJECT_DROP_DIR.glob("realised_*.wav"))


def rename_files_consistently():
    """Rename files to a consistent naming scheme."""
    
    wav_files = sorted(OBJECT_DROP_DIR.glob("realised_*.wav"))
    
    print(f"\n📝 Renaming {len(wav_files)} files for consistency...")
    
    for i, f in enumerate(wav_files, 1):
        new_name = f"object_drop_{i:04d}.wav"
        new_path = OBJECT_DROP_DIR / new_name
        
        if f.name != new_name:
            f.rename(new_path)
    
    final_files = list(OBJECT_DROP_DIR.glob("object_drop_*.wav"))
    print(f"✅ Renamed to: object_drop_0001.wav through object_drop_{len(final_files):04d}.wav")
    
    return final_files


def show_summary():
    """Show final summary of the dataset."""
    
    wav_files = list(OBJECT_DROP_DIR.glob("*.wav"))
    
    total_size = sum(f.stat().st_size for f in wav_files)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Location: {OBJECT_DROP_DIR}")
    print(f"Total WAV files: {len(wav_files)}")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")
    print(f"Format: 16-bit PCM WAV, 44.1kHz, Mono")
    print(f"Duration: ~7 seconds per file")
    print("="*60)


def main():
    print("="*60)
    print("ORGANIZING OBJECT DROP AUDIO DATASET")
    print("="*60)
    
    # Count current files
    count_files_by_category()
    
    # Note about the data
    print("\n" + "-"*60)
    print("NOTE: The ReaLISED dataset contains mixed indoor sounds.")
    print("We downloaded ALL 2479 files. The 'object falling' category")
    print("represents only ~190 of these files.")
    print("")
    print("Since we currently have 2479 mixed indoor sounds,")
    print("we can either:")
    print("  1. Keep all 2479 files (mixed indoor events)")
    print("  2. Keep a random sample of 600 files")
    print("")
    print("For PURE 'object falling' sounds, we should also")
    print("use the RealImpact dataset or Freesound.")
    print("-"*60)
    
    # Ask user preference (for now, keep 600)
    keep_random_sample(600)
    
    # Rename for consistency
    rename_files_consistently()
    
    # Show summary
    show_summary()


if __name__ == "__main__":
    main()
