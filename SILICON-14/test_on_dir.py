"""
SILICON-14: Model Validation on External 'test' Directory
Tests the trained model against raw background noises and new cry samples.
"""

import os
import numpy as np
import tensorflow as tf
import time
import librosa
from tqdm import tqdm
import config
from preprocess import ProPreprocessor
from model import TENetMicro, TENetBlock

def run_test_suite():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
    model_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}!")
        return
        
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found at {test_dir}!")
        return

    print("\n" + "="*70)
    print("      SILICON-14: EXTERNAL VALIDATION SUITE")
    print("="*70)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Test Folder: {test_dir}")
    print("-" * 70)

    # Load Model and Preprocessor
    from model import create_model
    model = create_model(num_classes=config.NUM_CLASSES)
    # Force build by calling on dummy data
    _ = model(np.zeros((1, 64, 148, 1)))
    model.load_weights(model_path)
    prep = ProPreprocessor()
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
    test_files.sort()
    
    results = []
    
    print(f"{'FILE NAME':<40} | {'PEAK PRED':<10} | {'DETECTIONS':<10}")
    print("-" * 70)
    
    for f in test_files:
        path = os.path.join(test_dir, f)
        
        try:
            # 1. Load full audio
            y, sr = librosa.load(path, sr=config.SAMPLE_RATE_HZ)
            
            # 2. SLIDING WINDOW INFERENCE (50% overlap)
            window_size = config.SAMPLE_LENGTH
            hop_size = window_size // 2
            
            file_detections = []
            max_conf = 0
            best_label = "Unknown"
            
            for start in range(0, len(y) - window_size, hop_size):
                window = y[start:start + window_size]
                
                # Apply the same Pro-Pipeline (NR, DC, etc.)
                # Since we don't want to re-load from disk, we'll manually call the steps
                window = prep._apply_dc_notch_filter(window)
                
                # Warm-up (Simulate real hardware buffer)
                noise_warmup = np.random.normal(0, 0.0008, config.WARMUP_SAMPLES)
                y_for_stft = np.concatenate([noise_warmup, window])
                
                # Extract Spectrogram
                stft = librosa.stft(y_for_stft, n_fft=512, hop_length=config.WINDOW_STEP_LENGTH, 
                                   win_length=config.WINDOW_SIZE_LENGTH, center=True)
                mel_basis = librosa.filters.mel(sr=config.SAMPLE_RATE_HZ, n_fft=512, n_mels=64, 
                                               fmin=config.FILTERBANK_LOWER_BAND_LIMIT, 
                                               fmax=config.FILTERBANK_UPPER_BAND_LIMIT)
                mel_spec = np.dot(mel_basis, np.abs(stft)**2)
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
                feat = mel_spec[:, -config.N_TIME_FRAMES:]
                
                # Predict
                feat_input = feat[np.newaxis, ..., np.newaxis]
                probs = model.predict(feat_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
                label = config.CLASS_NAMES[pred_idx]
                
                if label != "Unknown":
                    file_detections.append(label)
                
                if conf > max_conf:
                    max_conf = conf
                    best_label = label

            # 3. Aggregated Result
            summary = ", ".join(set(file_detections)) if file_detections else "None"
            display_name = (f[:37] + '..') if len(f) > 37 else f
            print(f"{display_name:<40} | {best_label:<10} | {summary}")
            
            results.append({
                'file': f,
                'label': best_label,
                'detections': set(file_detections)
            })
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("-" * 70)
    print("✅ TEST COMPLETE")
    
    # Analyze false positives (anything not baby/dog should be 'Unknown')
    noise_files = [r for r in results if any(k in r['file'].lower() for k in ['talking', 'restaurant', 'people', 'laugh'])]
    if noise_files:
        print("\nBackground Noise Robustness Check:")
        correct_unknown = sum(1 for r in noise_files if r['label'] == 'Unknown')
        print(f"  Accuracy on pure noise samples: {correct_unknown}/{len(noise_files)} ({correct_unknown/len(noise_files):.2%})")

    # Analyze catch rate on cries
    cry_files = [r for r in results if 'cry' in r['file'].lower()]
    if cry_files:
        print("\nBaby Cry Detection Check:")
        correct_cry = sum(1 for r in cry_files if r['label'] == 'Baby Cry')
        print(f"  Catch rate on test cries: {correct_cry}/{len(cry_files)} ({correct_cry/len(cry_files):.2%})")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    run_test_suite()
