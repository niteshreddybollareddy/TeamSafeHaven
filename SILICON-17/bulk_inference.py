"""
SILICON-17: PRODUCTION-GRADE BULK INFERENCE
- Implements Temporal Consistency (Streak Rule)
- Implements Multi-Window Aggregation
- Optimized for False Positive Rejection
- 3-Class: Baby Cry, Dog, Unknown
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
    print(f"      SILICON-17: PRODUCTION BULK INFERENCE")
    print(f"      Rule: Temporal Consistency (3-Window Streak)")
    print("="*60)

    # 1. Load Model
    model = create_model(input_shape=(64, 198, 1), num_classes=3)
    _ = model(np.zeros((1, 64, 198, 1))) 
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded.")
    else:
        print("⚠️ Warning: Model weights not found.")

    # 2. Setup Params
    prep = AFGPreprocessor()
    
    # Class-specific thresholds
    thresholds = {
        'baby_cry': 0.90, # High safety threshold
        'dog': 0.75,      # Balanced sensitivity
        'unknown': 0.0    # No threshold for catch-all
    }
    
    streak_required = 3  # Must have 3 consecutive 2.0s windows
    window_samples = 32000
    step_samples = 8000 # 0.5s step

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
            # Load and Resample manually via Preprocessor
            y, sr = sf.read(f_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            y = prep._manual_resample(y, sr, 16000)
            
            # --- TEMPORAL STATE MACHINE ---
            streaks = {'baby_cry': 0, 'dog': 0}
            max_streaks = {'baby_cry': 0, 'dog': 0}
            triggered = {'baby_cry': False, 'dog': False}
            
            # Slide window
            for start in range(0, len(y) - window_samples + 1, step_samples):
                window = y[start:start + window_samples]
                
                # Single window extraction
                feat = prep.extract_features(raw_audio=window)
                if feat is None:
                    # Reset streaks if window is too silent/inactive
                    for k in streaks: streaks[k] = 0
                    continue

                probs = model.predict(feat[np.newaxis, ..., np.newaxis], verbose=0)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
                label = CLASSES[pred_idx]

                # Streak Update: Only increment for the specific class IF confidence is high
                for class_name in ['baby_cry', 'dog']:
                    if label == class_name and conf >= thresholds[class_name]:
                        streaks[class_name] += 1
                        max_streaks[class_name] = max(max_streaks[class_name], streaks[class_name])
                        if streaks[class_name] >= streak_required:
                            triggered[class_name] = True
                    else:
                        streaks[class_name] = 0
                
            rel_path = os.path.relpath(f_path, target_dir)
            
            # Final Decision Priority Logic
            if triggered['baby_cry']:
                summary["detected_baby"] += 1
                print(f"\033[92m[👶 CRY] {rel_path:<40} | Streak: {max_streaks['baby_cry']}\033[0m")
            elif triggered['dog']: 
                summary["detected_dog"] += 1
                print(f"\033[94m[🐶 DOG] {rel_path:<40} | Streak: {max_streaks['dog']}\033[0m")
            # Else defaults to 'unknown' (implicit, no print)

        except Exception as e:
            continue

    # 5. Final Report
    print("\n" + "="*60)
    print("               FINAL BULK REPORT (SILICON-17)")
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
