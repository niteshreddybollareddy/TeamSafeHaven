"""
SILICON-17: FILE-BASED SLIDING WINDOW INFERENCE
Processes a WAV or MP3 file using a 2.0s sliding window.
3-Class: Baby Cry, Dog, Unknown
"""

import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from preprocess import AFGPreprocessor
from model import create_model

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_file_inference(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print("\n" + "="*60)
    print(f"      SILICON-18: FILE INFERENCE ({os.path.basename(file_path)})")
    print("="*60)

    # 1. Load Model
    model = create_model(input_shape=(64, 198, 1), num_classes=3)
    # Re-initialize/Build
    _ = model(np.zeros((1, 64, 198, 1))) 
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded.")
    else:
        print("⚠️ Warning: Model weights not found.")

    # 2. Load and Resample Audio
    print("⏳ Loading audio...")
    y, sr = sf.read(file_path, always_2d=False)
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    
    # 3. Setup Preprocessor
    prep = AFGPreprocessor()
    
    # Resample to 16kHz
    y = prep._manual_resample(y, sr, 16000)
    duration = len(y) / 16000
    print(f"✅ Audio loaded: {duration:.2f}s duration")
    
    # 4. Sliding Window Params
    # Window = 2.0s (32000 samples)
    # Step = 0.5s (8000 samples) for smooth detection
    window_samples = 32000
    step_samples = 8000 
    threshold = 0.70

    print("\n" + "-"*60)
    print(f"{'OFFSET (s)':<12} | {'PREDICTION':<15} | {'CONFIDENCE':<10}")
    print("-"*60)

    for start in range(0, len(y) - window_samples, step_samples):
        # Extract window
        window = y[start:start + window_samples]
        
        # 5. Preprocess (Handles DC Notch & Gate)
        feat = prep.extract_features(raw_audio=window)
        
        if feat is None:
            continue # Functionally silent or failed gate

        # 6. Inference
        feat_input = feat[np.newaxis, ..., np.newaxis]
        probs = model.predict(feat_input, verbose=0)[0]
        
        pred_idx = np.argmax(probs)
        conf = probs[pred_idx]
        label = CLASSES[pred_idx]
        offset_sec = start / 16000
        
        # 7. Terminal Output
        if conf > threshold and label != 'unknown':
            meter = "█" * int(conf * 10)
            color = "\033[92m" if label == 'baby_cry' else "\033[94m"
            icon = "👶" if label == 'baby_cry' else "🐶"
            desc = "CRY DETECTED" if label == 'baby_cry' else "DOG BARKING "
            print(f"{color}{offset_sec:05.1f}s - {offset_sec+2.0:0.1f}s | {icon} {desc} | {conf:.2%} {meter:<10}\033[0m")
        else:
             pass

    print("-" * 60)
    print("✅ Analysis Complete.")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 inference_file.py <path_to_audio_file>")
    else:
        run_file_inference(sys.argv[1])
