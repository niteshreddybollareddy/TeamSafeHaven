"""
SILICON-14: FIXED Model Validation on External 'test' Directory
KEY FIX: Uses ProPreprocessor.extract_features() for consistency
"""

import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tqdm import tqdm
import config
from preprocess import ProPreprocessor
from model import create_model

def run_test_suite_fixed():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
    model_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}!")
        return
        
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found at {test_dir}!")
        return

    print("\n" + "="*70)
    print("      SILICON-14: FIXED EXTERNAL VALIDATION SUITE")
    print("="*70)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Test Folder: {test_dir}")
    print("-" * 70)

    # Load Model
    model = create_model(num_classes=config.NUM_CLASSES)
    _ = model(np.zeros((1, 64, 148, 1)))  # Build model
    model.load_weights(model_path)
    
    # Load Preprocessor (CRITICAL: Use the same one as training!)
    prep = ProPreprocessor()
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
    test_files.sort()
    
    if not test_files:
        print("❌ No audio files found in test directory!")
        return
    
    results = []
    
    print(f"{'FILE NAME':<40} | {'PREDICTION':<12} | {'CONFIDENCE':<10} | {'ALL PROBS':<30}")
    print("-" * 70)
    
    for f in tqdm(test_files, desc="Testing"):
        path = os.path.join(test_dir, f)
        
        try:
            # METHOD 1: Single-window inference (for short files < 1.5s)
            # Use the EXACT same preprocessing as training
            feat = prep.extract_features(path, augment=False)
            
            # Predict
            feat_input = feat[np.newaxis, ..., np.newaxis]
            probs = model.predict(feat_input, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            label = config.CLASS_NAMES[pred_idx]
            
            # Format probabilities
            prob_str = ' '.join([f"{c[:3]}:{p:.2f}" for c, p in zip(config.CLASS_NAMES, probs)])
            
            display_name = (f[:37] + '..') if len(f) > 37 else f
            print(f"{display_name:<40} | {label:<12} | {confidence:.3f}      | {prob_str}")
            
            results.append({
                'file': f,
                'label': label,
                'confidence': confidence,
                'probs': probs
            })
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("-" * 70)
    print("✅ TEST COMPLETE\n")
    
    # Analysis
    analyze_results(results)

def run_sliding_window_test():
    """
    For longer audio files, use sliding window approach
    """
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
    model_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found!")
        return

    print("\n" + "="*70)
    print("      SILICON-14: SLIDING WINDOW TEST (For Long Files)")
    print("="*70)

    # Load Model
    model = create_model(num_classes=config.NUM_CLASSES)
    _ = model(np.zeros((1, 64, 148, 1)))
    model.load_weights(model_path)
    prep = ProPreprocessor()
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
    test_files.sort()
    
    print(f"{'FILE NAME':<40} | {'DETECTIONS':<30} | {'BEST PRED':<12}")
    print("-" * 70)
    
    for f in test_files:
        path = os.path.join(test_dir, f)
        
        try:
            # Load full audio
            y, sr = sf.read(path, always_2d=False)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            if sr != config.SAMPLE_RATE_HZ:
                y = librosa.resample(y, orig_sr=sr, target_sr=config.SAMPLE_RATE_HZ)
            
            # Save to temp file and process with ProPreprocessor in windows
            import tempfile
            window_size = config.SAMPLE_LENGTH
            hop_size = window_size // 2
            
            detections = []
            best_conf = 0
            best_label = "Unknown"
            
            # Sliding window
            for start in range(0, max(1, len(y) - window_size + 1), hop_size):
                window = y[start:start + window_size]
                
                # Save window to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, window, config.SAMPLE_RATE_HZ)
                    tmp_path = tmp.name
                
                # Use ProPreprocessor
                feat = prep.extract_features(tmp_path, augment=False)
                os.unlink(tmp_path)  # Clean up
                
                # Predict
                feat_input = feat[np.newaxis, ..., np.newaxis]
                probs = model.predict(feat_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
                label = config.CLASS_NAMES[pred_idx]
                
                if conf > config.DETECTION_THRESHOLD and label != "Unknown":
                    detections.append(label)
                
                if conf > best_conf:
                    best_conf = conf
                    best_label = label
            
            det_summary = ', '.join(set(detections)) if detections else "None"
            display_name = (f[:37] + '..') if len(f) > 37 else f
            print(f"{display_name:<40} | {det_summary:<30} | {best_label} ({best_conf:.2f})")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("-" * 70)

def analyze_results(results):
    """Analyze test results based on filename patterns"""
    
    # Categorize by filename
    baby_files = [r for r in results if any(k in r['file'].lower() for k in ['cry', 'baby', 'infant'])]
    dog_files = [r for r in results if any(k in r['file'].lower() for k in ['dog', 'bark', 'woof'])]
    noise_files = [r for r in results if any(k in r['file'].lower() for k in 
                   ['noise', 'background', 'talking', 'restaurant', 'people', 'laugh', 'traffic', 'music'])]
    
    print("📊 ANALYSIS BY FILE TYPE:")
    print("-" * 70)
    
    if baby_files:
        correct = sum(1 for r in baby_files if r['label'] == 'Baby Cry')
        avg_conf = np.mean([r['confidence'] for r in baby_files if r['label'] == 'Baby Cry'])
        print(f"👶 Baby Cry Files:")
        print(f"   Detected: {correct}/{len(baby_files)} ({correct/len(baby_files)*100:.1f}%)")
        if correct > 0:
            print(f"   Avg Confidence: {avg_conf:.3f}")
    
    if dog_files:
        correct = sum(1 for r in dog_files if r['label'] == 'Dog')
        avg_conf = np.mean([r['confidence'] for r in dog_files if r['label'] == 'Dog'])
        print(f"🐕 Dog Bark Files:")
        print(f"   Detected: {correct}/{len(dog_files)} ({correct/len(dog_files)*100:.1f}%)")
        if correct > 0:
            print(f"   Avg Confidence: {avg_conf:.3f}")
    
    if noise_files:
        correct = sum(1 for r in noise_files if r['label'] == 'Unknown')
        print(f"🔇 Background Noise Files:")
        print(f"   Correctly rejected: {correct}/{len(noise_files)} ({correct/len(noise_files)*100:.1f}%)")
        
        # Show false positives
        false_positives = [r for r in noise_files if r['label'] != 'Unknown']
        if false_positives:
            print(f"   ⚠️  False Positives:")
            for r in false_positives:
                print(f"      - {r['file']}: {r['label']} ({r['confidence']:.3f})")
    
    print("-" * 70)
    
    # Overall confidence distribution
    all_conf = [r['confidence'] for r in results]
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Mean Confidence: {np.mean(all_conf):.3f}")
    print(f"   Median Confidence: {np.median(all_conf):.3f}")
    print(f"   Std Confidence: {np.std(all_conf):.3f}")
    print(f"   High Confidence (>0.85): {sum(1 for c in all_conf if c > 0.85)}/{len(all_conf)}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sliding':
        run_sliding_window_test()
    else:
        run_test_suite_fixed()
        print("\nℹ️  For long audio files, run with: python test_on_dir_fixed.py --sliding")