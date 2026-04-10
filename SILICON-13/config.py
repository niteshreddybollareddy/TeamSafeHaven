"""
SILICON-13 Configuration
3-Class Classification: Baby Cry, Dog, Unknown
Based on SILICON-9 architecture - optimized for xG26 SoC with pure 2D CNN
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ============================================================================
# MLTK AudioFeatureGenerator Settings
# These settings match the hardware implementation on xG26 SoC
# ============================================================================

# Audio Input Settings
SAMPLE_RATE_HZ = 16000           # 16 kHz sampling (MLTK standard)
SAMPLE_LENGTH_MS = 1500          # 1.5 second audio clips
SAMPLE_LENGTH = int((SAMPLE_LENGTH_MS * SAMPLE_RATE_HZ) / 1000)  # 24000 samples
CHUNK_OVERLAP = 0.5              # 50% overlap for chunking long audio

# STFT (Spectrogram) Settings
WINDOW_SIZE_MS = 25              # 25 ms window (400 samples @ 16kHz)
WINDOW_STEP_MS = 10              # 10 ms hop (160 samples @ 16kHz)

# Mel Filterbank Settings
FILTERBANK_N_CHANNELS = 64       # 64 mel frequency bins
FILTERBANK_UPPER_BAND_LIMIT = 8000.0   # Nyquist = 8kHz for 16kHz SR
FILTERBANK_LOWER_BAND_LIMIT = 50.0     # Ignore below 50 Hz

# Noise Reduction Settings
NOISE_REDUCTION_ENABLE = True
NOISE_REDUCTION_SMOOTHING_BITS = 10
NOISE_REDUCTION_EVEN_SMOOTHING = 0.025
NOISE_REDUCTION_ODD_SMOOTHING = 0.06
NOISE_REDUCTION_MIN_SIGNAL_REMAINING = 0.40

# PCAN (Per-Channel Automatic Normalization)
PCAN_ENABLE = False

# Log Scaling
LOG_SCALE_ENABLE = True

# Activity Detection (VAD)
ACTIVITY_DETECTION_ENABLE = True
ACTIVITY_DETECTION_ALPHA_A = 0.5
ACTIVITY_DETECTION_ALPHA_B = 0.8
ACTIVITY_DETECTION_ARM_THRESHOLD = 0.75
ACTIVITY_DETECTION_TRIP_THRESHOLD = 0.8

# DC Notch Filter
DC_NOTCH_FILTER_ENABLE = True
DC_NOTCH_FILTER_COEFFICIENT = 0.95

# Dynamic Quantization
QUANTIZE_DYNAMIC_SCALE_ENABLE = True
QUANTIZE_DYNAMIC_SCALE_RANGE_DB = 40.0

# Calculated Values
WINDOW_SIZE_LENGTH = int((WINDOW_SIZE_MS * SAMPLE_RATE_HZ) / 1000)  # 400
WINDOW_STEP_LENGTH = int((WINDOW_STEP_MS * SAMPLE_RATE_HZ) / 1000)  # 160
N_TIME_FRAMES = (SAMPLE_LENGTH - WINDOW_SIZE_LENGTH) // WINDOW_STEP_LENGTH + 1  # 148
SPECTROGRAM_SHAPE = (FILTERBANK_N_CHANNELS, N_TIME_FRAMES)  # (64, 148)
FFT_LENGTH = 512

# ============================================================================
# Model Architecture Settings - CryNet3Class (based on CryNet-Micro-v2)
# ============================================================================

NUM_CLASSES = 3  # 3-Class: Baby Cry (0), Dog (1), Unknown (2)
MAX_PARAMS = 17000  # Budget for xG26

# CNN Architecture (same as SILICON-9)
CNN_FILTERS = [6, 12, 18, 24]  # Progressive channel expansion
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
USE_BATCH_NORM = True

# Classification Head
HEAD_UNITS = 16  # Slightly larger for 3 classes
DROPOUT = 0.4  # Strong regularization

# ============================================================================
# Training Settings
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

# Loss function
LOSS = 'categorical_crossentropy'
LABEL_SMOOTHING = 0.1  # Regularization

# Data split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class names - 3 classes
CLASS_NAMES = ['Baby Cry', 'Dog', 'Unknown']

# Class weights
USE_CLASS_WEIGHTS = True

# ============================================================================
# Data Augmentation Settings
# ============================================================================

USE_AUGMENTATION = True

# SpecAugment
USE_SPEC_AUGMENT = True
TIME_MASK_PARAM = 15  # Mask up to 15 time frames (~150ms)
FREQ_MASK_PARAM = 8   # Mask up to 8 mel bins
NUM_TIME_MASKS = 2
NUM_FREQ_MASKS = 1

# MixUp
USE_MIXUP = False
MIXUP_ALPHA = 0.4

# Audio augmentation (during preprocessing)
USE_AUDIO_AUGMENT = False
AUGMENT_PITCH_SHIFT = 2.0  # ±2 semitones
AUGMENT_TIME_STRETCH = 0.1  # ±10%
AUGMENT_NOISE_SNR = (20, 40)  # SNR range in dB

# Reproducibility
RANDOM_SEED = 42

# ============================================================================
# Expected Performance (xG26 SoC)
# ============================================================================

EXPECTED_PERFORMANCE = {
    'parameters': '~8,000',
    'model_size_kb': '~10',
    'inference_time_ms': '12-18',
    'ram_usage_kb': 16,
    'power_consumption_ma': 2.5,
    'accuracy': '>95%',
    'f1_score': '>0.94'
}

# ============================================================================
# TFLite Conversion Settings
# ============================================================================

TFLITE_QUANTIZE_INT8 = True
TFLITE_NUM_CALIBRATION_SAMPLES = 500

print(f"✓ Config loaded: SILICON-13 - 3-Class Audio Classifier")
print(f"  - Sample rate: {SAMPLE_RATE_HZ} Hz")
print(f"  - Sample length: {SAMPLE_LENGTH_MS} ms")
print(f"  - Spectrogram shape: {SPECTROGRAM_SHAPE}")
print(f"  - Model parameters budget: {MAX_PARAMS:,}")
print(f"  - Classes: {CLASS_NAMES}")
