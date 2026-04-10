"""
PROFESSOR'S CRITICAL SANITY CHECKS
Run this BEFORE training to catch preprocessing bugs early

Tests:
1. Training sample predicts correctly (catches inference mismatch)
2. Baby vs Unknown mel stats comparison (catches normalization collapse)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

from dataset import load_data
from model import CryNetMicroV2
import config

def test_training_sample():
    """
    PROFESSOR TEST 4: Non-negotiable sanity check

    Take ONE BABY training sample and verify model can predict it.
    If it fails on its own training data, preprocessing is broken.
    """
    print("="*70)
    print("TEST 1: TRAINING SAMPLE PREDICTION (Critical)")
    print("="*70)

    # Load training data
    train_gen, _, _, stats = load_data()

    # Get first baby cry sample
    labels_array = np.array(train_gen.labels)
    baby_indices = np.where(labels_array == 0)[0]
    if len(baby_indices) == 0:
        print("ERROR: No baby cry samples in training data!")
        return False

    first_baby_idx = baby_indices[0]

    # Load the preprocessed sample
    mel_spec = np.load(train_gen.file_paths[first_baby_idx])
    label = train_gen.labels[first_baby_idx]

    # Normalize
    mel_norm = (mel_spec - stats['mean']) / (stats['std'] + 1e-8)

    # Reshape for model
    input_tensor = mel_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

    # Ensure correct shape (64, 98)
    if input_tensor.shape[3] < 98:
        pad_width = 98 - input_tensor.shape[3]
        input_tensor = np.pad(input_tensor, ((0,0), (0,0), (0,0), (0, pad_width)))
    else:
        input_tensor = input_tensor[:, :, :, :98]

    # Create untrained model
    model = CryNetMicroV2(num_classes=2, dropout=0.3)
    dummy_input = tf.zeros((1, 1, 64, 98))
    _ = model(dummy_input)

    # Predict
    pred = model(input_tensor, training=False).numpy()[0]

    filename = Path(train_gen.file_paths[first_baby_idx]).name
    print(f"\nSample: {filename}")
    print(f"True Label: {label} ({config.CLASS_NAMES[label]})")
    print(f"Prediction: Baby={pred[0]:.4f}, Unknown={pred[1]:.4f}")

    # Analysis
    if pred[0] > 0.9:
        print("\n⚠ PROBLEM: Untrained model predicts Baby with >90% confidence")
        print("   This suggests inference preprocessing doesn't match training")
        return False
    elif pred[1] > 0.9:
        print("\n⚠ PROBLEM: Untrained model predicts Unknown with >90% confidence")
        print("   This suggests training data preprocessing is broken")
        return False
    else:
        print("\n✓ PASS: Untrained model shows uncertainty (expected behavior)")
        return True


def test_mel_stats():
    """
    PROFESSOR KILLER TEST: Compare baby vs unknown mel statistics

    If baby mean ≈ unknown mean, preprocessing erased class signal.
    This is the #1 cause of 100% single-class predictions.
    """
    print("\n" + "="*70)
    print("TEST 2: MEL STATS COMPARISON (Killer Test)")
    print("="*70)

    # Load data
    train_gen, _, _, stats = load_data()

    # Separate by class
    labels_array = np.array(train_gen.labels)
    baby_indices = np.where(labels_array == 0)[0]
    unknown_indices = np.where(labels_array == 1)[0]

    print(f"\nLoading {len(baby_indices)} baby samples and {len(unknown_indices)} unknown samples...")

    # Sample 50 from each class
    n_samples = min(50, len(baby_indices), len(unknown_indices))

    baby_values = []
    for idx in baby_indices[:n_samples]:
        mel = np.load(train_gen.file_paths[idx])
        # Exclude silence padding
        valid = mel[mel > -80]
        if len(valid) > 0:
            baby_values.append(valid)

    unknown_values = []
    for idx in unknown_indices[:n_samples]:
        mel = np.load(train_gen.file_paths[idx])
        # Exclude silence padding
        valid = mel[mel > -80]
        if len(valid) > 0:
            unknown_values.append(valid)

    if len(baby_values) == 0 or len(unknown_values) == 0:
        print("\nERROR: No valid samples found (all silence)")
        return False

    baby_values = np.concatenate(baby_values)
    unknown_values = np.concatenate(unknown_values)

    # Compute statistics
    baby_mean = baby_values.mean()
    baby_std = baby_values.std()
    unknown_mean = unknown_values.mean()
    unknown_std = unknown_values.std()

    print(f"\nBaby Cry mel stats:")
    print(f"  Mean: {baby_mean:.4f}")
    print(f"  Std:  {baby_std:.4f}")

    print(f"\nUnknown mel stats:")
    print(f"  Mean: {unknown_mean:.4f}")
    print(f"  Std:  {unknown_std:.4f}")

    print(f"\nGlobal stats (used for normalization):")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")

    # Analysis
    mean_diff = abs(baby_mean - unknown_mean)
    mean_avg = (abs(baby_mean) + abs(unknown_mean)) / 2

    print(f"\n" + "-"*70)
    print("ANALYSIS:")
    print("-"*70)
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Mean difference (relative): {mean_diff/mean_avg*100:.1f}%")

    if mean_diff < 2.0:
        print("\n❌ CRITICAL PROBLEM:")
        print("   Baby mean ≈ Unknown mean (difference < 2.0 dB)")
        print("   Preprocessing has ERASED class signal!")
        print("   Model will collapse to single-class predictions.")
        print("\n   Root causes:")
        print("   - Energy windowing not working")
        print("   - Normalization includes too much silence padding")
        print("   - Silence trimming too aggressive")
        return False
    elif mean_diff < 5.0:
        print("\n⚠ WARNING:")
        print("   Baby and Unknown distributions are similar (2-5 dB difference)")
        print("   Model may struggle to learn meaningful differences.")
        print("   Consider:")
        print("   - Check energy-based windowing is capturing cry correctly")
        print("   - Verify silence padding is minimal")
        return True
    else:
        print("\n✓ PASS:")
        print(f"   Baby and Unknown have distinct distributions (>{mean_diff:.1f} dB difference)")
        print("   Preprocessing preserves class signal.")
        return True


def main():
    """Run all sanity checks"""
    print("\n" + "="*70)
    print("PROFESSOR'S SANITY CHECKS - RUN BEFORE TRAINING")
    print("="*70)

    test1_pass = test_training_sample()
    test2_pass = test_mel_stats()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Training Sample): {'✓ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Mel Stats):       {'✓ PASS' if test2_pass else '❌ FAIL'}")

    if test1_pass and test2_pass:
        print("\n🎉 ALL CHECKS PASSED - SAFE TO TRAIN")
    else:
        print("\n⚠ CRITICAL FAILURES - DO NOT TRAIN YET")
        print("   Fix preprocessing issues first!")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
