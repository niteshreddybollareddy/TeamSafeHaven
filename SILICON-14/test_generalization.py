"""
MODEL GENERALIZATION TEST
Tests if the model learned real patterns or just memorized training data
"""

import os
import numpy as np
import tensorflow as tf
import config
from model import create_model
from preprocess import ProPreprocessor

def test_model_generalization():
    """Test if model makes reasonable predictions on synthetic data"""
    prep = ProPreprocessor()
    
    print("="*70)
    print("MODEL GENERALIZATION TEST")
    print("="*70)
    
    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    model = create_model(num_classes=config.NUM_CLASSES)
    _ = model(np.zeros((1, 64, 148, 1)))
    model.load_weights(model_path)
    
    print("\n1. TESTING ON RANDOM NOISE")
    print("-"*70)
    
    # Test 1: Random noise (should predict Unknown with low confidence)
    normalized_noise = np.random.randn(5, 64, 148, 1) * 0.1
    preds = model.predict(normalized_noise, verbose=0)
    
    for i, pred in enumerate(preds):
        pred_idx = np.argmax(pred)
        label = config.CLASS_NAMES[pred_idx]
        conf = pred[pred_idx]
        print(f"  Random Noise {i+1}: {label} (confidence: {conf:.3f})")
    
    avg_conf = np.mean([np.max(p) for p in preds])
    print(f"\n  Average confidence on noise: {avg_conf:.3f}")
    
    if avg_conf > 0.8:
        print("  ❌ PROBLEM: Model is TOO confident on random noise!")
        print("     This suggests overfitting or poor generalization")
    elif avg_conf > 0.5:
        print("  ⚠️  Model is somewhat confident on noise (not ideal)")
    else:
        print("  ✅ Model appropriately uncertain on noise")
    
    print("\n2. TESTING ON BLANK/SILENT INPUT")
    print("-"*70)
    
    # Test 2: Silent input (should predict Unknown)
    normalized_silent = np.ones((5, 64, 148, 1)) * -80.0  # -80dB is silence in log scale
    preds = model.predict(normalized_silent, verbose=0)
    
    for i, pred in enumerate(preds):
        pred_idx = np.argmax(pred)
        label = config.CLASS_NAMES[pred_idx]
        conf = pred[pred_idx]
        print(f"  Silent Input {i+1}: {label} (confidence: {conf:.3f})")
    
    unknown_count = sum([1 for p in preds if np.argmax(p) == config.CLASS_NAMES.index('Unknown')])
    print(f"\n  {unknown_count}/5 predicted as Unknown")
    
    if unknown_count < 3:
        print("  ❌ PROBLEM: Model predicts Baby/Dog on silence!")
        print("     This is a serious issue - model hasn't learned what silence is")
    else:
        print("  ✅ Model correctly handles silent input")
    
    print("\n3. TESTING MODEL CONFIDENCE CALIBRATION")
    print("-"*70)
    
    # Test 3: Check if probabilities are calibrated
    # Load some real training samples
    test_samples = []
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        if os.path.exists(class_dir):
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('_clean.npy')][:10]
            for npy_file in npy_files:
                feat = np.load(os.path.join(class_dir, npy_file))
                test_samples.append(feat[np.newaxis, ..., np.newaxis])
    
    if test_samples:
        all_preds = []
        for sample in test_samples:
            pred = model.predict(sample, verbose=0)[0]
            all_preds.append(pred)
        
        max_probs = [np.max(p) for p in all_preds]
        avg_max_prob = np.mean(max_probs)
        
        print(f"  Average max probability on training samples: {avg_max_prob:.3f}")
        print(f"  Std of max probabilities: {np.std(max_probs):.3f}")
        
        over_confident = sum([1 for p in max_probs if p > 0.99])
        print(f"  {over_confident}/{len(max_probs)} predictions with >99% confidence")
        
        if avg_max_prob > 0.95:
            print("\n  ⚠️  Model is VERY confident (might be overfit)")
        else:
            print("\n  ✅ Model has reasonable confidence levels")
    
    print("\n4. FEATURE IMPORTANCE CHECK")
    print("-"*70)
    
    # Test 4: See if model uses all frequency bands
    # Create input with only low frequencies
    low_freq_only = np.random.randn(1, 64, 148, 1) * 0.1
    low_freq_only[:, 32:, :, :] = -80.0  # Mask upper frequencies
    
    # Create input with only high frequencies  
    high_freq_only = np.random.randn(1, 64, 148, 1) * 0.1
    high_freq_only[:, :32, :, :] = -80.0  # Mask lower frequencies
    
    pred_low = model.predict(low_freq_only, verbose=0)[0]
    pred_high = model.predict(high_freq_only, verbose=0)[0]
    
    print(f"  Low frequencies only: {config.CLASS_NAMES[np.argmax(pred_low)]} ({np.max(pred_low):.3f})")
    print(f"  High frequencies only: {config.CLASS_NAMES[np.argmax(pred_high)]} ({np.max(pred_high):.3f})")
    
    print("\n5. TESTING MODEL WEIGHTS")
    print("-"*70)
    
    # Test 5: Check if weights are reasonable
    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")
    
    # Get first layer weights
    first_layer_weights = model.layers[0].get_weights()[0] if len(model.layers[0].get_weights()) > 0 else None
    
    if first_layer_weights is not None:
        print(f"  First layer weight mean: {np.mean(first_layer_weights):.6f}")
        print(f"  First layer weight std: {np.std(first_layer_weights):.6f}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(first_layer_weights)) or np.any(np.isinf(first_layer_weights)):
            print("  ❌ CRITICAL: Model has NaN or Inf weights!")
        else:
            print("  ✅ Weights are valid")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nIf the model:")
    print("  • Is too confident on noise → Overfitting")
    print("  • Predicts Baby/Dog on silence → Hasn't learned features properly")
    print("  • Has >99% confidence on everything → Over-regularized or memorizing")
    print("\nThese issues would cause real-world test failures!")

if __name__ == '__main__':
    test_model_generalization()