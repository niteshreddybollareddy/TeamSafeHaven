"""
SILICON-15: DATA PREPARATION SCRIPT
- Reads raw audio from Dataset/
- Converts to MLTK-style Spectrograms (64, 148)
- Saves as .npy files in processed_dataset/
"""

import os
import numpy as np
from tqdm import tqdm
from preprocess import AFGPreprocessor

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'unknown']

def prepare():
    print("\n" + "="*50)
    print("      SILICON-15: DATA PREPROCESSING (MLTK SIMULATION)")
    print("="*50)
    
    prep = AFGPreprocessor()
    
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
    
    # Create debug folder to verify "Max Energy Crop"
    DEBUG_AUDIO_PATH = os.path.join(BASE_DIR, 'debug_audio')
    os.makedirs(DEBUG_AUDIO_PATH, exist_ok=True)
    import soundfile as sf

    for label in CLASSES:
        source_dir = os.path.join(DATASET_PATH, label)
        output_dir = os.path.join(PROCESSED_PATH, label)
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(source_dir) if f.endswith(('.wav', '.mp3'))]
        print(f"\n🚀 Processing {label} ({len(files)} files)...")
        
        for idx, f_name in enumerate(tqdm(files)):
            try:
                f_path = os.path.join(source_dir, f_name)
                f_stem = os.path.splitext(f_name)[0]
                
                # 1. BASE VERSION (Clean) - Get audio too for 5% of files
                should_debug = (idx % 20 == 0)
                if should_debug:
                    feat_clean, audio_window = prep.extract_features(f_path, augment=False, return_audio=True)
                    sf.write(os.path.join(DEBUG_AUDIO_PATH, f"{label}_{f_stem}_crop.wav"), audio_window, 16000)
                else:
                    feat_clean = prep.extract_features(f_path, augment=False)
                
                np.save(os.path.join(output_dir, f"{f_stem}_clean.npy"), feat_clean)
                
                # 2. MLTK AUGMENTATIONS (Generate 2 different versions per file)
                for i in range(2):
                    feat_aug = prep.extract_features(f_path, augment=True)
                    np.save(os.path.join(output_dir, f"{f_stem}_aug_{i}.npy"), feat_aug)
                    
            except Exception as e:
                print(f"  ❌ Error processing {f_name}: {e}")

    print("\n✅ PREPROCESSING COMPLETE")
    print(f"📂 Processed data saved to: {PROCESSED_PATH}")
    print("="*50 + "\n")

if __name__ == '__main__':
    prepare()
