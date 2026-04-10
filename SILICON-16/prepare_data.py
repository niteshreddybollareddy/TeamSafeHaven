"""
SILICON-16: DATA PREPARATION (V3)
- Strictly enforces 40% active signal via Preprocessor V3.
- Skips files that fail the activity/energy checks.
- Balances 'unknown' by sampling multiple active windows from long files.
"""

import os
import numpy as np
from tqdm import tqdm
from preprocess import AFGPreprocessor
import soundfile as sf

BASE_DIR = "/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-16"
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'unknown']

def prepare():
    print("\n" + "="*60)
    print("      SILICON-16: STRICT DATA PREPARATION (V3)")
    print("="*60)
    
    prep = AFGPreprocessor()
    if not os.path.exists(PROCESSED_PATH): os.makedirs(PROCESSED_PATH)

    stats = {cls: {'kept': 0, 'rejected': 0} for cls in CLASSES}

    for label in CLASSES:
        source_dir = os.path.join(DATASET_PATH, label)
        output_dir = os.path.join(PROCESSED_PATH, label)
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(source_dir) if f.endswith(('.wav', '.mp3', '.ogg', '.m4a'))]
        print(f"\n🚀 Checking {label} signal quality ({len(files)} source files)...")
        
        for f_name in tqdm(files):
            f_path = os.path.join(source_dir, f_name)
            f_stem = os.path.splitext(f_name)[0]
            
            try:
                info = sf.info(f_path)
                # Strategic sampling: Multiple windows from long unknown sources
                num_windows = 1
                if label == 'unknown':
                    if info.duration > 15: num_windows = 3
                    elif info.duration > 8: num_windows = 2

                valid_windows_in_file = 0
                for i in range(num_windows):
                    result = prep.extract_features(audio_path=f_path, label=label, return_audio=True)
                    
                    if result is None:
                        continue # Skip this window (too quiet or too much silence)
                    
                    feat, audio = result
                    valid_windows_in_file += 1
                    
                    suffix = f"_{i}" if num_windows > 1 else ""
                    np.save(os.path.join(output_dir, f"{f_stem}{suffix}_clean.npy"), feat)
                    sf.write(os.path.join(output_dir, f"{f_stem}{suffix}_clean.wav"), audio, 16000)
                    
                    # Generate 2 Augmented versions per valid window
                    for aug_idx in range(2):
                        feat_aug, audio_aug = prep.extract_features(audio_path=f_path, label=label, augment=True, return_audio=True)
                        if feat_aug is not None:
                            np.save(os.path.join(output_dir, f"{f_stem}{suffix}_aug_{aug_idx}.npy"), feat_aug)
                            sf.write(os.path.join(output_dir, f"{f_stem}{suffix}_aug_{aug_idx}.wav"), audio_aug, 16000)

                if valid_windows_in_file > 0:
                    stats[label]['kept'] += 1
                else:
                    stats[label]['rejected'] += 1
                    
            except Exception:
                stats[label]['rejected'] += 1
                continue

    print("\n" + "-"*60)
    print("               PREPROCESSING RESULTS")
    print("-"*60)
    for cls in CLASSES:
        print(f"[{cls.upper()}] Kept: {stats[cls]['kept']} | Rejected: {stats[cls]['rejected']} (Low Signal)")
    print(f"\n📂 Clean data in: {PROCESSED_PATH}")
    print("="*60 + "\n")

if __name__ == '__main__':
    prepare()
