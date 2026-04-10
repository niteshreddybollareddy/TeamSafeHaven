"""
SILICON-20: PCEN CONFIDENCE & PRECISION VALIDATOR
- Analyzes the trained model to find optimal decision thresholds.
- Applies Z-score normalization from saved stats.
- 3-Class: Baby Cry, Dog, Unknown
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
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_validation():
    print("\n" + "="*60)
    print("      SILICON-20: PCEN CONFIDENCE VALIDATOR")
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

    # 2. Load Normalization Stats
    if os.path.exists(STATS_PATH):
        stats = np.load(STATS_PATH, allow_pickle=True).item()
        norm_mean = stats['mean']
        norm_std = stats['std']
        # Extract scalars for easier broadcasting
        mean_val = norm_mean.item() if hasattr(norm_mean, 'item') else norm_mean
        std_val = norm_std.item() if hasattr(norm_std, 'item') else norm_std
        print("✅ PCEN Normalization Stats Loaded.")
    else:
        mean_val, std_val = 0, 1
        print("⚠️ Warning: Normalization stats not found. Using defaults (0, 1).")

    results = {label: [] for label in CLASSES}

    # 3. Collect RAW Probabilities
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir): 
            print(f"⚠️ Warning: {label} processed directory not found.")
            continue
        
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not files: 
            print(f"⚠️ Warning: No .npy files found for {label}.")
            continue
        
        # Take a subset if the dataset is huge (more than 500)
        subset_count = min(500, len(files))
        subset_files = np.random.choice(files, subset_count, replace=False)
        
        print(f"🧐 Analyzing {label} ({subset_count} samples)...")
        for f in tqdm(subset_files):
            # Load feature
            feat = np.load(f)
            # Reshape to (64, 198, 1)
            feat = feat[..., np.newaxis]
            
            # Apply Normalization
            feat = (feat - mean_val) / (std_val + 1e-9)
            
            # Prediction (add batch dim)
            probs = model.predict(feat[np.newaxis, ...], verbose=0)[0]
            results[label].append(probs)

    # 4. Analyze Separation
    print("\n" + "-"*60)
    print("               DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    thresholds = [0.5, 0.65, 0.8, 0.9, 0.95, 0.98]
    
    for t in thresholds:
        print(f"\n[ THRESHOLD: {t} ]")
        
        for target_idx, target_label in enumerate(['baby_cry', 'dog']):
            # True Positives
            tp = sum(1 for p in results[target_label] if p[target_idx] >= t)
            total_samples = len(results[target_label])
            recall = tp / total_samples if total_samples > 0 else 0
            
            # False Positives from 'unknown'
            fp_unknown = sum(1 for p in results['unknown'] if p[target_idx] >= t)
            
            # False Positives from other target class
            other_label = 'dog' if target_label == 'baby_cry' else 'baby_cry'
            fp_other = sum(1 for p in results[other_label] if p[target_idx] >= t)
            
            total_fp = fp_unknown + fp_other
            precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0
            
            print(f"   {target_label.upper():<10} | Precision: {precision:.2%} | Recall: {recall:.2%} | False Triggers: {total_fp}/{len(results['unknown']) + len(results[other_label])}")

    print("\n" + "="*60)
    print("💡 SILICON-20 Recommendation:")
    print("   Select a threshold for 'baby_cry' that maximizes recall while keeping precision > 95%.")
    print("   PCEN usually allows for lower thresholds (e.g. 0.65) with high precision.")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_validation()
