"""
SILICON DUAL-MODEL STRESS TEST (S20 vs S21)
- Compares SILICON-20 (PCEN V1) vs SILICON-21 (CryNet Architecture)
- Dataset: SILICON-1/Dataset/cry
- Logic: Master Policy V1 (Prioritized Decaying Streaks)
"""

import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from tqdm import tqdm
import importlib.util

# Paths
BASE_PATH = "/Users/theepireddykarthikreddy/Desktop/SILICON"
S20_DIR = os.path.join(BASE_PATH, "silicon-20 pcen")
S21_DIR = os.path.join(BASE_PATH, "silicon-21 pcen architecture")
DATASET_PATH = "/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-1/Dataset/cry"

def load_module_from_dir(dir_path, module_name):
    file_path = os.path.join(dir_path, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"{module_name}_{os.path.basename(dir_path)}", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_stress_test():
    print("\n" + "="*80)
    print("🚀 SILICON DUAL-MODEL STRESS TEST (S20 PCEN vs S21 CryNet)")
    print("="*80)

    # 1. Load Models and their respective Preprocessors
    print("📦 Loading Models and Preprocessors...")
    
    # S20 Setup
    mod20 = load_module_from_dir(S20_DIR, "model")
    prep20_mod = load_module_from_dir(S20_DIR, "preprocess")
    model_20 = mod20.create_model(input_shape=(64, 198, 1), num_classes=3)
    model_20.load_weights(os.path.join(S20_DIR, "best_model.h5"))
    prep20 = prep20_mod.AFGPreprocessor()
    
    # S21 Setup
    mod21 = load_module_from_dir(S21_DIR, "model")
    prep21_mod = load_module_from_dir(S21_DIR, "preprocess")
    model_21 = mod21.create_model(input_shape=(64, 198, 1), num_classes=3)
    # Using by_name and skip_mismatch for CryNet inference loading
    model_21.load_weights(os.path.join(S21_DIR, "best_model.h5"), by_name=True, skip_mismatch=True)
    prep21 = prep21_mod.AFGPreprocessor()
    
    # Load Normalization Stats
    def load_stats(dir_path):
        stats_path = os.path.join(dir_path, "normalization_stats.npy")
        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True).item()
            return stats['mean'].item(), stats['std'].item()
        return 0, 1

    s20_mean, s20_std = load_stats(S20_DIR)
    s21_mean, s21_std = load_stats(S21_DIR)

    print("✅ SILICON-20 Loaded (PCEN V1).")
    print("✅ SILICON-21 Loaded (CryNet Architecture).")

    # Policy Parameters (Standard Master Policy V1)
    thresholds = {'baby_cry': 0.65, 'dog': 0.90}
    required_streaks = {'baby_cry': 2, 'dog': 4}
    decay_rates = {'baby_cry': 1.0, 'dog': 0.3}
    baby_suppression_floor = 0.35
    
    window_samples = 32000
    step_samples = 4000 # 250ms step (Real-time simulation)

    all_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) 
                 if f.lower().endswith(('.wav', '.ogg', '.mp3'))]
    
    print(f"🔍 Analyzing {len(all_files)} files...\n")

    stats = {
        "S20": {"detections": 0, "false_dog": 0, "false_unknown": 0},
        "S21": {"detections": 0, "false_dog": 0, "false_unknown": 0}
    }

    for f_path in tqdm(all_files, desc="Stress Testing"):
        try:
            y, sr = sf.read(f_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            y16 = prep21._manual_resample(y, sr, 16000) # S21 resampler is same
            
            # Reset PCEN states
            prep20.reset_pcen_state()
            prep21.reset_pcen_state()
            
            # Temporal State Engines
            streaks = {"S20": {'baby_cry': 0.0, 'dog': 0.0}, "S21": {'baby_cry': 0.0, 'dog': 0.0}}
            trig_baby = {"S20": False, "S21": False}
            trig_dog = {"S20": False, "S21": False}
            
            # Slide window
            for start in range(0, len(y16) - window_samples + 1, step_samples):
                window = y16[start:start + window_samples]
                
                # --- S20 Path ---
                feat20 = prep20.extract_features(raw_audio=window, is_inference=True)
                if feat20 is not None:
                    feat20_norm = (feat20[..., np.newaxis] - s20_mean) / (s20_std + 1e-9)
                    p20 = model_20.predict(feat20_norm[np.newaxis, ...], verbose=0)[0]
                    
                    # S20 uses basic streak logic
                    if p20[0] >= thresholds['baby_cry']:
                        streaks["S20"]['baby_cry'] += 1
                        streaks["S20"]['dog'] = max(0, streaks["S20"]['dog'] - decay_rates['dog'])
                    elif p20[1] >= thresholds['dog']:
                        streaks["S20"]['dog'] += 1
                        streaks["S20"]['baby_cry'] = max(0, streaks["S20"]['baby_cry'] - decay_rates['baby_cry'])
                    else:
                        for k in streaks["S20"]: streaks["S20"][k] = max(0, streaks["S20"][k] - decay_rates[k])

                # --- S21 Path (CryNet + Enhanced Policy) ---
                feat21 = prep21.extract_features(raw_audio=window, is_inference=True)
                if feat21 is not None:
                    # Silence check (Hard Energy Gate)
                    rms = np.sqrt(np.mean(window**2))
                    if rms < 0.002:
                        for k in streaks["S21"]: streaks["S21"][k] = max(0, streaks["S21"][k] - decay_rates[k])
                    else:
                        feat21_norm = (feat21[..., np.newaxis] - s21_mean) / (s21_std + 1e-9)
                        p21 = model_21.predict(feat21_norm[np.newaxis, ...], verbose=0)[0]
                        baby_prob, dog_prob = p21[0], p21[1]

                        # S21 Master Policy V1
                        if max(baby_prob, dog_prob) < 0.7:
                            for k in streaks["S21"]: streaks["S21"][k] = max(0, streaks["S21"][k] - decay_rates[k])
                        elif baby_prob >= thresholds['baby_cry']:
                            streaks["S21"]['baby_cry'] += 1
                            streaks["S21"]['dog'] = max(0, streaks["S21"]['dog'] - decay_rates['dog'])
                        elif (dog_prob >= thresholds['dog'] and (dog_prob - baby_prob) > 0.15 and baby_prob < baby_suppression_floor):
                            streaks["S21"]['dog'] += 1
                            streaks["S21"]['baby_cry'] = max(0, streaks["S21"]['baby_cry'] - decay_rates['baby_cry'])
                        else:
                            for k in streaks["S21"]: streaks["S21"][k] = max(0, streaks["S21"][k] - decay_rates[k])

                # Check triggers
                for m in ["S20", "S21"]:
                    if streaks[m]['baby_cry'] >= required_streaks['baby_cry']: trig_baby[m] = True
                    elif streaks[m]['dog'] >= required_streaks['dog']: trig_dog[m] = True

            # Finalize Tally
            for m in ["S20", "S21"]:
                if trig_baby[m]: stats[m]["detections"] += 1
                elif trig_dog[m]: stats[m]["false_dog"] += 1
                else: stats[m]["false_unknown"] += 1

        except Exception:
            continue

    # Final Summary Report
    total = stats["S20"]["detections"] + stats["S20"]["false_dog"] + stats["S20"]["false_unknown"]
    print("\n" + "="*80)
    print(f"📊 STRESS TEST REPORT: SILICON-20 vs SILICON-21 ({total} total)")
    print("="*80)
    
    for m in ["S20", "S21"]:
        name = "SILICON-20 (PCEN V1)" if m == "S20" else "SILICON-21 (CryNet MTL)"
        hits = stats[m]["detections"]
        acc = (hits/total)*100 if total > 0 else 0
        print(f"\n{name}")
        print(f"Detections (Cry) : {hits}")
        print(f"False Dog        : {stats[m]['false_dog']}")
        print(f"False Unknown    : {stats[m]['false_unknown']}")
        print(f"Recall Accuracy  : {acc:.2f}%")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    run_stress_test()
