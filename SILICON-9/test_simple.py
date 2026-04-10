"""
SILICON-9 Simple Test Script
Test files in /Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-9/test directory
Output: filename and probabilities for each class

PREPROCESSING APPLIED (matches training exactly):
✓ DC notch filter (removes DC offset)
✓ Silence trimming (top_db=30)
✓ Energy-based windowing (ALWAYS - captures loudest part for both classes)
✓ Pre-emphasis filter (boosts high frequencies)
✓ Mel-spectrogram extraction (64 mel bins, 98 time frames)
✓ Normalization with training mean/std

OPTION 3 IMPLEMENTATION:
- BOTH classes use energy-based windowing during training
- Inference uses same energy-based windowing (no ambiguity)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
sys.path.append('/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-9')
import config
from preprocess import AudioPreprocessor
from model import CryNetMicroV2

# HARDCODED PATHS - UPDATED for new training
TEST_DIR = Path('/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-9/test - files')
WEIGHTS_PATH = Path('/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-9/experiments/crynet_micro_v2_20260111_182431/checkpoints/best_model.weights.h5')
STATS_PATH = Path('/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-9/experiments/crynet_micro_v2_20260111_182431/checkpoints/stats.npy')

# NEW NORMALIZATION STATS (from professor fixes training)
MEAN = -31.3581
STD = 12.7735


def test_directory():
    """Test all audio files in hardcoded test directory"""

    print("="*70)
    print("SILICON-9 SIMPLE TESTER")
    print("="*70)
    print(f"Test directory: {TEST_DIR}")
    print(f"Weights:        {WEIGHTS_PATH}")
    print(f"Stats:          {STATS_PATH}")
    print("="*70 + "\n")

    # Use hardcoded normalization stats (updated from new training)
    mean = MEAN
    std = STD
    print(f"Using normalization: mean={mean:.4f}, std={std:.4f}\n")

    # Build model architecture
    print("Building model architecture...")
    model = CryNetMicroV2(num_classes=2, dropout=0.3)

    # Build model with dummy input
    dummy_input = tf.zeros((1, 1, 64, 98))
    _ = model(dummy_input)

    # Load weights
    print("Loading weights...")
    model.load_weights(WEIGHTS_PATH)
    print("✓ Model ready\n")

    # Initialize preprocessor
    preprocessor = AudioPreprocessor()

    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg', '*.flac', '*.webm']:
        audio_files.extend(list(TEST_DIR.glob(ext)))

    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"⚠ No audio files found in {TEST_DIR}")
        return

    print(f"Found {len(audio_files)} audio files\n")
    print("="*70)
    print(f"{'Filename':<40} {'Baby Cry %':<12} {'Unknown %':<12}")
    print("="*70)

    # Test each file
    for audio_file in audio_files:
        try:
            # OPTION 3: Always use energy-based windowing (no filename detection needed)
            mel_spec = preprocessor.preprocess_file(
                str(audio_file),
                return_raw=True,
                debug=False
            )

            # Normalize
            mel_spec_norm = (mel_spec - mean) / (std + 1e-8)

            # Reshape: (64, T) -> (1, 1, 64, T)
            input_tensor = mel_spec_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

            # Pad/trim to expected frames
            expected_frames = 98
            if input_tensor.shape[3] < expected_frames:
                pad_width = expected_frames - input_tensor.shape[3]
                input_tensor = np.pad(input_tensor, ((0,0), (0,0), (0,0), (0, pad_width)))
            else:
                input_tensor = input_tensor[:, :, :, :expected_frames]

            # Predict
            output = model.predict(input_tensor, verbose=0)[0]

            baby_cry_prob = output[0] * 100
            unknown_prob = output[1] * 100

            # Print result
            print(f"{audio_file.name:<40} {baby_cry_prob:>10.2f}%  {unknown_prob:>10.2f}%")

        except Exception as e:
            print(f"{audio_file.name:<40} ERROR: {str(e)}")

    print("="*70)
    print("\nTesting complete!")


if __name__ == '__main__':
    test_directory()
