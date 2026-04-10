"""
SILICON-16: PRODUCTION-GRADE BULK INFERENCE
- Implements Temporal Consistency (Streak Rule)
- Implements Multi-Window Aggregation
- Optimized for False Positive Rejection
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm
from preprocess import AFGPreprocessor
from model import create_binary_model

# Config
BASE_DIR = "/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-16"
MODEL_PATH = os.path.join(BASE_DIR, 'best_binary_model.h5')

def run_bulk_inference(target_dir):
    if not os.path.exists(target_dir):
        print(f"❌ Directory not found: {target_dir}")
        return

    print("\n" + "="*60)
    print(f"      SILICON-16: PRODUCTION BULK INFERENCE")
    print(f"      Rule: Temporal Consistency (3-Window Streak)")
    print("="*60)

    # 1. Load Model
    model = create_binary_model(input_shape=(64, 198, 1), num_classes=2)
    _ = model(np.zeros((1, 64, 198, 1))) 
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded.")

    # 2. Setup Params
    prep = AFGPreprocessor()
    threshold = 0.70
    streak_required = 3  # Must have 3 consecutive 2.0s windows
    window_samples = 32000
    step_samples = 8000 # 0.5s step = high resolution temporal tracking

    # 3. Find files
    audio_extensions = ('.wav', '.mp3', '.ogg', '.m4a')
    all_files = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(audio_extensions):
                all_files.append(os.path.join(root, f))
    
    print(f"🔍 Found {len(all_files)} audio files. Analyzing...\n")

    summary = {"total": 0, "detected": 0}

    for f_path in tqdm(all_files, desc="Processing"):
        try:
            summary["total"] += 1
            y, sr = librosa.load(f_path, sr=16000, mono=True)
            
            # --- TEMPORAL STATE MACHINE ---
            current_streak = 0
            max_streak = 0
            confidences = []
            file_triggered = False
            
            # Slide window
            for start in range(0, len(y) - window_samples + 1, step_samples):
                window = y[start:start + window_samples]
                
                # Preprocess (Standard logic, no label-bias)
                feat = prep.extract_features(raw_audio=window)
                if feat is None:
                    current_streak = 0 # Reset streak on silence
                    continue

                probs = model.predict(feat[np.newaxis, ..., np.newaxis], verbose=0)[0]
                conf = probs[0] # Index 0 = cry
                confidences.append(conf)

                if conf >= threshold:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
                
                # TRIGGER RULE: 3 consecutive windows
                if current_streak >= streak_required:
                    file_triggered = True
                    # We continue scanning to get full stats, or break for speed
                    # break 

            if file_triggered:
                summary["detected"] += 1
                median_conf = np.median([c for c in confidences if c > 0.01]) if confidences else 0
                rel_path = os.path.relpath(f_path, target_dir)
                print(f"\033[92m[👶 TRIGGER] {rel_path:<40} | Streak: {max_streak} | Med-Conf: {median_conf:.2%}\033[0m")

        except Exception as e:
            # print(f"Error {f_path}: {e}")
            continue

    # 5. Final Report
    print("\n" + "="*60)
    print("               FINAL BULK REPORT")
    print("-" * 60)
    print(f"Total Files Processed   : {summary['total']}")
    print(f"Production-Level Triggers : {summary['detected']}")
    print(f"False Discovery Rate (Est): {(summary['detected']/summary['total'])*100:.2f}%")
    print("="*60 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 bulk_inference.py <directory_path>")
    else:
        run_bulk_inference(sys.argv[1])
