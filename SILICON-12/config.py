"""
SILICON-12 Configuration
3-Class Classification: Baby Cry, Dog Barking, Unknown
Based on SILICON-9 architecture (proven 99.81% accuracy) extended to 3 classes

KEY DESIGN DECISIONS (Why SILICON-9 worked, SILICON-10/11 failed):
1. Use SILICON-9's proven architecture (7,160 params baseline)
2. Same preprocessing pipeline (energy-based windowing)
3. Same 1.5s window (98 time frames) - proven to work
4. LayerNorm for stable single-sample inference
5. Single-window inference (no multi-window averaging that dilutes signal)
6. Standard Conv2D (not depthwise separable - need expressiveness)
7. Dense(16) intermediate layer before output (CRITICAL for discrimination)

CRITICAL: Real-time inference accuracy requires EXACT match between
training and inference preprocessing. No shortcuts.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
EXPERIMENT_DIR = os.path.join(BASE_DIR, 'experiments')

# ============================================================================
# AUDIO INPUT SETTINGS (SILICON-9 PROVEN VALUES)
# ============================================================================

SAMPLE_RATE_HZ = 16000           # 16 kHz sampling (standard)
SAMPLE_LENGTH_MS = 1500          # 1.5 second audio clips (SILICON-9 proven)
SAMPLE_LENGTH = int((SAMPLE_LENGTH_MS * SAMPLE_RATE_HZ) / 1000)  # 24000 samples

# ============================================================================
# STFT (Spectrogram) Settings (SILICON-9 PROVEN VALUES)
# ============================================================================

WINDOW_SIZE_MS = 25              # 25 ms window (400 samples @ 16kHz)
WINDOW_STEP_MS = 10              # 10 ms hop (160 samples @ 16kHz) - SILICON-9 value
                                 # DO NOT use 5ms (SILICON-10's mistake - too many frames)

# Mel Filterbank Settings
FILTERBANK_N_CHANNELS = 64       # 64 mel frequency bins
FILTERBANK_UPPER_BAND_LIMIT = 8000.0   # Nyquist = 8kHz for 16kHz SR
FILTERBANK_LOWER_BAND_LIMIT = 50.0     # Ignore below 50 Hz

# Calculated Values
WINDOW_SIZE_LENGTH = int((WINDOW_SIZE_MS * SAMPLE_RATE_HZ) / 1000)  # 400
WINDOW_STEP_LENGTH = int((WINDOW_STEP_MS * SAMPLE_RATE_HZ) / 1000)  # 160
FFT_LENGTH = 512

# Time frames calculation:
# With center=True (librosa default): adds FFT_LENGTH//2 padding on each side
# n_frames = 1 + (n_samples + n_fft - win_length) / hop_length
# = 1 + (24000 + 512 - 400) / 160 = 152 frames
#
# With center=False: n_frames = (n_samples - win_length) // hop_length + 1
# = (24000 - 400) // 160 + 1 = 148 frames
#
# We use center=False for predictable frame count
N_TIME_FRAMES = (SAMPLE_LENGTH - WINDOW_SIZE_LENGTH) // WINDOW_STEP_LENGTH + 1  # 148
SPECTROGRAM_SHAPE = (FILTERBANK_N_CHANNELS, N_TIME_FRAMES)  # (64, 148)

# ============================================================================
# PREPROCESSING SETTINGS (SILICON-9 PROVEN VALUES)
# ============================================================================

# Noise Reduction (light, preserves signal)
NOISE_REDUCTION_ENABLE = True
NOISE_REDUCTION_SMOOTHING_BITS = 10
NOISE_REDUCTION_EVEN_SMOOTHING = 0.025
NOISE_REDUCTION_ODD_SMOOTHING = 0.06
NOISE_REDUCTION_MIN_SIGNAL_REMAINING = 0.40

# DC Notch Filter (removes DC offset)
DC_NOTCH_FILTER_ENABLE = True
DC_NOTCH_FILTER_COEFFICIENT = 0.95

# Silence Trimming (LESS aggressive than SILICON-10)
SILENCE_TRIM_TOP_DB = 40  # SILICON-9 uses 40, SILICON-10 used 30 (too aggressive)

# Pre-emphasis
PREEMPHASIS_COEFF = 0.97

# ============================================================================
# ENERGY-BASED WINDOWING (CRITICAL FOR REAL-TIME ACCURACY)
# ============================================================================

# SILICON-9's key insight: Always use max-energy window for ALL classes
# This ensures the model sees the loudest/most important part of audio
# Applied BEFORE silence trimming to preserve signal onset

ENERGY_WINDOW_FRAME_LENGTH = 1024  # RMS calculation frame
ENERGY_WINDOW_HOP_LENGTH = 512     # RMS calculation hop

# ============================================================================
# MODEL ARCHITECTURE (SILICON-9 PROVEN + EXTENDED TO 3 CLASSES)
# ============================================================================

NUM_CLASSES = 3  # Baby Cry, Dog Barking, Unknown
CLASS_NAMES = ['baby cry', 'dog', 'unknown']  # Must match Dataset folder names

# CNN Architecture (SILICON-9 proven values - standard Conv2D, NOT depthwise)
CNN_FILTERS = [6, 12, 18, 24]  # Progressive channel expansion (SILICON-9)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
USE_DEPTHWISE_SEPARABLE = False  # CRITICAL: Use standard Conv2D for expressiveness

# Classification Head (SILICON-9 has Dense intermediate layer - CRITICAL)
HEAD_UNITS = 16  # Slightly more than SILICON-9's 12 for 3-class discrimination
DROPOUT = 0.4    # Strong regularization (SILICON-9 value)

# Parameter budget
MAX_PARAMS = 10000  # xG26 budget

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 150
PATIENCE = 20
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# Learning rate schedule
USE_COSINE_SCHEDULE = True
MIN_LR = 1e-5
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-4

# Loss function
LOSS = 'categorical_crossentropy'
LABEL_SMOOTHING = 0.1  # Regularization

# Data split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class weights (will be computed from dataset)
USE_CLASS_WEIGHTS = True

# ============================================================================
# DATA AUGMENTATION (During training only)
# ============================================================================

USE_AUGMENTATION = True

# SpecAugment
USE_SPEC_AUGMENT = True
TIME_MASK_PARAM = 15  # Mask up to 15 time frames (~150ms)
FREQ_MASK_PARAM = 8   # Mask up to 8 mel bins
NUM_TIME_MASKS = 2
NUM_FREQ_MASKS = 1

# MixUp
USE_MIXUP = True
MIXUP_ALPHA = 0.4

# Reproducibility
RANDOM_SEED = 42

# ============================================================================
# TFLITE CONVERSION
# ============================================================================

TFLITE_QUANTIZE_INT8 = True
TFLITE_NUM_CALIBRATION_SAMPLES = 500

# ============================================================================
# EXPECTED PERFORMANCE
# ============================================================================

EXPECTED_PERFORMANCE = {
    'parameters': '~7,500',
    'model_size_kb': '~8',
    'inference_time_ms': '12-18',
    'ram_usage_kb': '~20',
    'target_accuracy': '>95%',
    'target_auc': '>0.98'
}

# ============================================================================
# REAL-TIME INFERENCE SETTINGS
# ============================================================================

# CRITICAL: These must match training EXACTLY
INFERENCE_SETTINGS = {
    'window_selection': 'max_energy',  # NOT sliding, NOT random
    'aggregation': 'none',              # Single window, no averaging
    'normalization': 'global_zscore',   # Use training mean/std
    'confidence_threshold': 0.5,        # For binary decisions
}

# ============================================================================
# MLTK Integration (for xG26 deployment)
# ============================================================================

MLTK_MODEL_PARAMETERS = {
    'fe.sample_rate_hz': SAMPLE_RATE_HZ,
    'fe.sample_length_ms': SAMPLE_LENGTH_MS,
    'fe.window_size_ms': WINDOW_SIZE_MS,
    'fe.window_step_ms': WINDOW_STEP_MS,
    'fe.filterbank_n_channels': FILTERBANK_N_CHANNELS,
    'fe.filterbank_upper_band_limit': FILTERBANK_UPPER_BAND_LIMIT,
    'fe.filterbank_lower_band_limit': FILTERBANK_LOWER_BAND_LIMIT,
    'fe.noise_reduction_enable': NOISE_REDUCTION_ENABLE,
    'fe.dc_notch_filter_enable': DC_NOTCH_FILTER_ENABLE,
    'fe.dc_notch_filter_coefficient': DC_NOTCH_FILTER_COEFFICIENT,
    'fe.fft_length': FFT_LENGTH
}

# ============================================================================
# VERIFICATION
# ============================================================================

def verify_config():
    """Verify configuration is consistent"""
    expected_frames = (SAMPLE_LENGTH - WINDOW_SIZE_LENGTH) // WINDOW_STEP_LENGTH + 1
    assert expected_frames == N_TIME_FRAMES, f"Frame mismatch: {expected_frames} vs {N_TIME_FRAMES}"
    assert N_TIME_FRAMES == 148, f"Expected 148 frames, got {N_TIME_FRAMES}"
    assert SPECTROGRAM_SHAPE == (64, 148), f"Unexpected spectrogram shape: {SPECTROGRAM_SHAPE}"
    return True

# Run verification on import
try:
    verify_config()
    print(f"SILICON-12 Config loaded: 3-class ({CLASS_NAMES})")
    print(f"  Sample: {SAMPLE_LENGTH_MS}ms | Spectrogram: {SPECTROGRAM_SHAPE}")
    print(f"  Architecture: SILICON-9 base (standard Conv2D, LayerNorm)")
except AssertionError as e:
    print(f"CONFIG ERROR: {e}")
    raise
