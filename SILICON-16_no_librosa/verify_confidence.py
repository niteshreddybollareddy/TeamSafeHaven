"""
SILICON-16: CONFIDENCE & PRECISION VALIDATOR
Analyzes the trained model to find the optimal decision threshold.
"""

import os
import numpy as np
import tensorflow as tf
from model import create_binary_model
from tqdm import tqdm

# Config
BASE_DIR = "/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-16"
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'best_binary_model.h5')
CLASSES = ['baby_cry', 'unknown']

def run_validation():
    print("\n" + "="*60)
    print("      SILICON-16: CONFIDENCE VALIDATOR")
    print("="*60)

    # 1. Load Model
    model = create_binary_model(input_shape=(64, 198, 1), num_classes=2)
    # Build graph
    _ = model(np.zeros((1, 64, 198, 1))) 
    model.load_weights(MODEL_PATH)
    print("Model loaded.")

    results = {cls: [] for cls in CLASSES}

    # 2. Collect RAW Probabilities
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        # Take a subset if the dataset is huge (e.g., 500 samples per class)
        subset_files = np.random.choice(files, min(500, len(files)), replace=False)
        
        print(f"🧐 Analyzing {label} (500 samples)...")
        for f in tqdm(subset_files):
            feat = np.load(f)[np.newaxis, ..., np.newaxis]
            # Probabilities for [baby_cry, unknown]
            probs = model.predict(feat, verbose=0)[0]
            # We care about the 'baby_cry' probability (index 0)
            cry_prob = probs[0]
            results[label].append(cry_prob)

    # 3. Analyze Separation
    print("\n" + "-"*60)
    print("               DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    thresholds = [0.5, 0.7, 0.9, 0.95, 0.98]
    
    for t in thresholds:
        print(f"\n[ THRESHOLD: {t} ]")
        
        # True Positives: Baby cry samples correctly identified
        tp = sum(1 for p in results['baby_cry'] if p >= t)
        fn = len(results['baby_cry']) - tp
        recall = tp / len(results['baby_cry'])
        
        # False Positives: Unknown samples incorrectly identified as cry
        fp = sum(1 for p in results['unknown'] if p >= t)
        tn = len(results['unknown']) - fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"   Precision (Cry): {precision:.2%}")
        print(f"   Recall (Cry)   : {recall:.2%}")
        print(f"   False Triggers : {fp} out of {len(results['unknown'])}")


if __name__ == '__main__':
    run_validation()
