"""
SILICON-16: FILE-BASED SLIDING WINDOW INFERENCE
Processes a WAV or MP3 file using a 2.0s sliding window.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from preprocess import AFGPreprocessor
from model import create_binary_model

# Config
BASE_DIR = "/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-16"
MODEL_PATH = os.path.join(BASE_DIR, 'best_binary_model.h5')

def run_file_inference(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print("\n" + "="*60)
    print(f"      SILICON-16: FILE INFERENCE ({os.path.basename(file_path)})")
    print("="*60)

    # 1. Load Model
    model = create_binary_model(input_shape=(64, 198, 1), num_classes=2)
    # Re-initialize/Build
    _ = model(np.zeros((1, 64, 198, 1))) 
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded.")

    # 2. Load and Resample Audio
    print("⏳ Loading audio...")
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    duration = len(y) / 16000
    print(f"✅ Audio loaded: {duration:.2f}s duration")

    # 3. Setup Preprocessor
    prep = AFGPreprocessor()
    
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
        
        # Index 0 = baby_cry
        cry_confidence = probs[0]
        offset_sec = start / 16000
        
        # 7. Terminal Output
        if cry_confidence > threshold:
            meter = "█" * int(cry_confidence * 10)
            print(f"\033[92m{offset_sec:05.1f}s - {offset_sec+2.0:0.1f}s | 👶 CRY DETECTED | {cry_confidence:.2%} {meter:<10}\033[0m")
        elif cry_confidence > 0.3:
            # Subtle logging for background
             # print(f"{offset_sec:05.1f}s - {offset_sec+2.0:0.1f}s | ... listening    | {cry_confidence:.2%}")
             pass
        else:
             # Very low confidence - keep quiet
             pass

    print("-" * 60)
    print("✅ Analysis Complete.")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 inference_file.py <path_to_audio_file>")
    else:
        run_file_inference(sys.argv[1])
