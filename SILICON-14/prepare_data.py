"""
SILICON-14: Pure Data Preparation
- Source: SILICON-10
- Balancing: Perfect 1:1:1 Ratio
- Processing: Energy-Windowing for bursts and Ambient Padding for short files
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config
from preprocess import ProPreprocessor
import random

def get_balanced_file_list():
    """Find the smallest class size and balance all others to it."""
    file_list = {}
    min_size = float('inf')
    
    for class_name, folder_name in config.CLASS_MAPPING.items():
        class_path = Path(config.SOURCE_DIR) / folder_name
        if not class_path.exists():
            print(f"⚠ Warning: {class_path} not found")
            continue
        
        files = list(class_path.glob('*.wav'))
        file_list[class_name] = files
        min_size = min(min_size, len(files))
        
    print(f"Balanced Class Size: {min_size} samples per class")
    
    # Trim to min_size
    balanced_list = {}
    for class_name, files in file_list.items():
        np.random.seed(42)
        np.random.shuffle(files)
        balanced_list[class_name] = files[:min_size]
        
    return balanced_list

def process_pure_dataset():
    prep = ProPreprocessor()
    balanced_data = get_balanced_file_list()
    processed_dir = Path(config.PROCESSED_DIR)
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print("\n🚀 Processing Balanced Dataset with MLTK Augmentations...")
    
    for class_name, files in balanced_data.items():
        class_output = processed_dir / class_name
        os.makedirs(class_output, exist_ok=True)
        
        for f in tqdm(files, desc=f"Processing {class_name}"):
            try:
                # 1. BASE VERSION (No Augmentation - Clean reference)
                feat_clean = prep.extract_features(str(f), augment=False)
                np.save(class_output / f"{f.stem}_clean.npy", feat_clean)
                
                # 2. MLTK VARIATIONS (2 augmented copies per file)
                for i in range(2):
                    feat_aug = prep.extract_features(str(f), augment=True)
                    np.save(class_output / f"{f.stem}_aug{i}.npy", feat_aug)
                
                # THE PRO TRICK: Create "Partial-Event" awareness for the Unknown class
                if class_name in ['Baby Cry', 'Dog']:
                    unknown_out = processed_dir / 'Unknown'
                    os.makedirs(unknown_out, exist_ok=True)
                    
                    # We take the clean feat and slash its time dimension
                    partial_feat = feat_clean.copy()
                    cut_idx = config.N_TIME_FRAMES // 3
                    partial_feat[:, cut_idx:] = -80.0 
                    
                    np.save(unknown_out / f"partial_{f.stem}.npy", partial_feat)
                    
            except Exception as e:
                print(f"Error processing {f.name}: {e}")

    print("\n✅ Dataset Ready: Balanced 1:1:1")
    print(f"Location: {config.PROCESSED_DIR}")
    print("="*60 + "\n")

if __name__ == '__main__':
    process_pure_dataset()
