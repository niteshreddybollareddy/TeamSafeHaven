"""
SILICON-19: PRODUCTION-GRADE BULK INFERENCE (8kHz INT8)
- Exact SL Hardware Parity
- Implements Master Policy V1 (Prioritized Decaying Streaks)
"""

import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from tqdm import tqdm
from preprocess import AFGPreprocessor
from model import create_model

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_bulk_inference(target_dir):
    if not os.path.exists(target_dir):
        print(f"❌ Directory not found: {target_dir}")
        return

    print("\n" + "="*60)
    print(f"      SILICON-19: PRODUCTION BULK INFERENCE")
    print(f"      Rule: Master Policy V1 (Exact SL Logic)")
    print("="*60)

    # 1. Load Model (Updated for SILICON-19)
    model = create_model(input_shape=(32, 198, 1), num_classes=3)
    _ = model(np.zeros((1, 32, 198, 1))) 
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded.")
    else:
        print("⚠️ Warning: Model weights not found.")

    # 2. Setup Params
    prep = AFGPreprocessor()
    thresholds = {'baby_cry': 0.65, 'dog': 0.92}
    required_streaks = {'baby_cry': 2, 'dog': 4}
    
    window_samples = 16000 # 2.0s @ 8kHz
    step_samples = 4000   # 0.5s step @ 8kHz

    # 3. Find files
    audio_extensions = ('.wav', '.mp3', '.ogg', '.m4a')
    all_files = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(audio_extensions):
                all_files.append(os.path.join(root, f))
    
    print(f"🔍 Found {len(all_files)} audio files. Analyzing...\n")

    summary = {"total": 0, "detected_baby": 0, "detected_dog": 0}

    for f_path in tqdm(all_files, desc="Processing"):
        try:
            summary["total"] += 1
            y, sr = sf.read(f_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            y = prep._manual_resample(y, sr, 16000)
            
            # --- TEMPORAL STATE MACHINE (MASTER POLICY V1) ---
            streaks = {'baby_cry': 0, 'dog': 0}
            triggered = {'baby_cry': False, 'dog': False}
            
            # Slide window
            for start in range(0, len(y) - window_samples + 1, step_samples):
                window = y[start:start + window_samples]
                feat = prep.extract_features(raw_audio=window)
                
                if feat is None:
                    for k in streaks: streaks[k] = max(0, streaks[k] - 1)
                    continue

                probs = model.predict(feat[np.newaxis, ..., np.newaxis], verbose=0)[0]
                
                # Priority Decision Logic
                if probs[0] >= thresholds['baby_cry']:
                    streaks['baby_cry'] += 1
                    streaks['dog'] = max(0, streaks['dog'] - 1)
                elif probs[1] >= thresholds['dog']:
                    streaks['dog'] += 1
                    streaks['baby_cry'] = max(0, streaks['baby_cry'] - 1)
                else:
                    for k in streaks: streaks[k] = max(0, streaks[k] - 1)

                if streaks['baby_cry'] >= required_streaks['baby_cry']: triggered['baby_cry'] = True
                if streaks['dog'] >= required_streaks['dog']: triggered['dog'] = True
                
            rel_path = os.path.relpath(f_path, target_dir)
            if triggered['baby_cry']:
                summary["detected_baby"] += 1
                print(f"\033[92m[👶 CRY] {rel_path:<40}\033[0m")
            elif triggered['dog']: 
                summary["detected_dog"] += 1
                print(f"\033[94m[🐶 DOG] {rel_path:<40}\033[0m")

        except Exception:
            continue

    # 5. Final Report
    print("\n" + "="*60)
    print("               FINAL BULK REPORT (SILICON-19)")
    print("-" * 60)
    print(f"Total Files Processed   : {summary['total']}")
    print(f"Baby Cry Detections     : {summary['detected_baby']}")
    print(f"Dog Bark Detections     : {summary['detected_dog']}")
    print("="*60 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 bulk_inference.py <directory_path>")
    else:
        run_bulk_inference(sys.argv[1])
