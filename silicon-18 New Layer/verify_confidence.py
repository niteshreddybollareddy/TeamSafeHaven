"""
SILICON-17: CONFIDENCE & PRECISION VALIDATOR
Analyzes the trained model to find the optimal decision threshold.
3-Class: Baby Cry, Dog, Unknown
"""

import os
import numpy as np
import tensorflow as tf
from model import create_model
from tqdm import tqdm

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_validation():
    print("\n" + "="*60)
    print("      SILICON-18: CONFIDENCE VALIDATOR")
    print("="*60)

    # 1. Load Model
    model = create_model(input_shape=(64, 198, 1), num_classes=3)
    # Build graph
    _ = model(np.zeros((1, 64, 198, 1))) 
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded.")
    else:
        print("⚠️ Warning: Model weights not found.")
        return

    results = {label: [] for label in CLASSES}

    # 2. Collect RAW Probabilities
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir): continue
        
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not files: continue
        
        # Take a subset if the dataset is huge
        subset_count = min(500, len(files))
        subset_files = np.random.choice(files, subset_count, replace=False)
        
        print(f"🧐 Analyzing {label} ({subset_count} samples)...")
        for f in tqdm(subset_files):
            feat = np.load(f)[np.newaxis, ..., np.newaxis]
            probs = model.predict(feat, verbose=0)[0]
            results[label].append(probs)

    # 3. Analyze Separation
    print("\n" + "-"*60)
    print("               DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    thresholds = [0.5, 0.7, 0.9, 0.95, 0.98]
    
    for t in thresholds:
        print(f"\n[ THRESHOLD: {t} ]")
        
        for target_idx, target_label in enumerate(['baby_cry', 'dog']):
            # True Positives
            tp = sum(1 for p in results[target_label] if p[target_idx] >= t)
            recall = tp / len(results[target_label]) if results[target_label] else 0
            
            # False Positives from 'unknown'
            fp_unknown = sum(1 for p in results['unknown'] if p[target_idx] >= t)
            
            # False Positives from other target class
            other_label = 'dog' if target_label == 'baby_cry' else 'baby_cry'
            fp_other = sum(1 for p in results[other_label] if p[target_idx] >= t)
            
            total_fp = fp_unknown + fp_other
            precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0
            
            print(f"   {target_label.upper():<10} | Precision: {precision:.2%} | Recall: {recall:.2%} | False Triggers: {total_fp}")

    print("\n" + "="*60)
    print("💡 Recommendation:")
    print("   Look for the threshold where 'False Triggers' drops to near zero.")
    print("   High precision is critical for edge deployment.")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_validation()
