"""
SILICON-8 Configuration
Using Silicon Labs MLTK AudioFeatureGenerator
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
# See: https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
# ============================================================================

# Audio Input Settings
SAMPLE_RATE_HZ = 16000           # 16 kHz sampling (standard for speech/audio)
SAMPLE_LENGTH_MS = 1000          # 1 second audio clips
SAMPLE_LENGTH = int((SAMPLE_LENGTH_MS * SAMPLE_RATE_HZ) / 1000)  # 16000 samples

# STFT (Spectrogram) Settings
WINDOW_SIZE_MS = 25              # 25 ms window (400 samples @ 16kHz)
WINDOW_STEP_MS = 10              # 10 ms hop (160 samples @ 16kHz)
                                 # Results in ~97 time frames per 1-second clip

# Mel Filterbank Settings
FILTERBANK_N_CHANNELS = 64       # 64 mel frequency bins (mel bands)
FILTERBANK_UPPER_BAND_LIMIT = 8000.0   # Upper freq limit (Nyquist = 8kHz for 16kHz SR)
FILTERBANK_LOWER_BAND_LIMIT = 50.0     # Lower freq limit (ignore below 50 Hz)

# Noise Reduction Settings (improves robustness to background noise)
NOISE_REDUCTION_ENABLE = True
NOISE_REDUCTION_SMOOTHING_BITS = 10
NOISE_REDUCTION_EVEN_SMOOTHING = 0.025
NOISE_REDUCTION_ODD_SMOOTHING = 0.06
NOISE_REDUCTION_MIN_SIGNAL_REMAINING = 0.40  # Preserve 40% of signal

# PCAN (Per-Channel Automatic Normalization) - Optional
PCAN_ENABLE = False              # Disabled for now, enable if needed
PCAN_STRENGTH = 0.95
PCAN_OFFSET = 80.0
PCAN_GAIN_BITS = 21

# Log Scaling (convert filterbank output to log scale)
LOG_SCALE_ENABLE = True          # Essential for audio classification
LOG_SCALE_SHIFT = 6              # Bit shift for log scaling

# Activity Detection (Voice Activity Detection)
ACTIVITY_DETECTION_ENABLE = True  # Helps filter out silence
ACTIVITY_DETECTION_ALPHA_A = 0.5  # Fast filter coefficient
ACTIVITY_DETECTION_ALPHA_B = 0.8  # Slow filter coefficient
ACTIVITY_DETECTION_ARM_THRESHOLD = 0.75   # Threshold to arm detection
ACTIVITY_DETECTION_TRIP_THRESHOLD = 0.8   # Threshold to trip detection

# DC Notch Filter (removes DC offset)
DC_NOTCH_FILTER_ENABLE = True    # Recommended for real-world audio
DC_NOTCH_FILTER_COEFFICIENT = 0.95

# Dynamic Quantization (for INT8 output)
QUANTIZE_DYNAMIC_SCALE_ENABLE = True   # Enable for TFLite INT8 deployment
QUANTIZE_DYNAMIC_SCALE_RANGE_DB = 40.0 # 40 dB dynamic range

# Calculated Values
# These are computed automatically by MLTK based on above settings
WINDOW_SIZE_LENGTH = int((WINDOW_SIZE_MS * SAMPLE_RATE_HZ) / 1000)  # 400
WINDOW_STEP_LENGTH = int((WINDOW_STEP_MS * SAMPLE_RATE_HZ) / 1000)  # 160
N_TIME_FRAMES = (SAMPLE_LENGTH - WINDOW_SIZE_LENGTH) // WINDOW_STEP_LENGTH + 1  # ~97
SPECTROGRAM_SHAPE = (N_TIME_FRAMES, FILTERBANK_N_CHANNELS)  # (97, 64)

# FFT length (automatically calculated - smallest power of 2 >= window size)
FFT_LENGTH = 512  # Next power of 2 after 400

# ============================================================================
# Model Architecture Settings
# ============================================================================

NUM_CLASSES = 4
MAX_PARAMS = 17000

# Transformer architecture (same as SILICON-7)
D_MODEL = 18
NUM_HEADS = 3
NUM_LAYERS = 3
DROPOUT = 0.15

# Squeeze-and-Excitation block
USE_SE_BLOCK = True
SE_REDUCTION = 3

# Multi-scale CNN configuration
# Using 3x3 (3 filters) + 5x5 (2 filters) + 7x7 (1 filter) = 6 total
MULTISCALE_CNN_FILTERS = {
    '3x3': 3,
    '5x5': 2,
    '7x7': 1
}

# ============================================================================
# Training Settings
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 250
PATIENCE = 25
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# Learning rate scheduling
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_START_LR = 3e-5
USE_COSINE_SCHEDULE = True

# Focal Loss parameters (better than standard cross-entropy)
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class weights (if needed, but Focal Loss handles imbalance)
USE_CLASS_WEIGHTS = False

# Class names
CLASS_NAMES = ['Baby Cry', 'Dog', 'Unknown', 'Unusual']

# Data Augmentation (applied during training)
USE_AUGMENTATION = True
TIME_MASK_PARAM = 40       # SpecAugment time masking
FREQ_MASK_PARAM = 12       # SpecAugment frequency masking
NUM_MASKS = 2
MIXUP_ALPHA = 0.4          # MixUp augmentation

# Reproducibility
RANDOM_SEED = 42

# ============================================================================
# Production Notes
# ============================================================================

PRODUCTION_IMPROVEMENTS = [
    "Uses Silicon Labs MLTK AudioFeatureGenerator (hardware-optimized)",
    "Same preprocessing code runs on xG26 SoC (no porting needed)",
    "Activity detection for VAD (Voice Activity Detection)",
    "Dynamic quantization for INT8 deployment",
    "Noise reduction for robust real-world performance",
    "DC notch filter removes DC offset",
]

EXPECTED_PERFORMANCE = {
    'accuracy': '91-92%',
    'inference_time_ms': '50-100',
    'model_size_kb': 15.5,
    'power_consumption_ma': 3.3,
    'parameters': 15531
}

# ============================================================================
# MLTK Integration
# ============================================================================

# These settings will be added to the TFLite model metadata
# so the xG26 device knows how to preprocess audio
MLTK_MODEL_PARAMETERS = {
    'fe.sample_rate_hz': SAMPLE_RATE_HZ,
    'fe.sample_length_ms': SAMPLE_LENGTH_MS,
    'fe.window_size_ms': WINDOW_SIZE_MS,
    'fe.window_step_ms': WINDOW_STEP_MS,
    'fe.filterbank_n_channels': FILTERBANK_N_CHANNELS,
    'fe.filterbank_upper_band_limit': FILTERBANK_UPPER_BAND_LIMIT,
    'fe.filterbank_lower_band_limit': FILTERBANK_LOWER_BAND_LIMIT,
    'fe.noise_reduction_enable': NOISE_REDUCTION_ENABLE,
    'fe.noise_reduction_smoothing_bits': NOISE_REDUCTION_SMOOTHING_BITS,
    'fe.noise_reduction_even_smoothing': NOISE_REDUCTION_EVEN_SMOOTHING,
    'fe.noise_reduction_odd_smoothing': NOISE_REDUCTION_ODD_SMOOTHING,
    'fe.noise_reduction_min_signal_remaining': NOISE_REDUCTION_MIN_SIGNAL_REMAINING,
    'fe.pcan_enable': PCAN_ENABLE,
    'fe.pcan_strength': PCAN_STRENGTH,
    'fe.pcan_offset': PCAN_OFFSET,
    'fe.pcan_gain_bits': PCAN_GAIN_BITS,
    'fe.log_scale_enable': LOG_SCALE_ENABLE,
    'fe.log_scale_shift': LOG_SCALE_SHIFT,
    'fe.activity_detection_enable': ACTIVITY_DETECTION_ENABLE,
    'fe.activity_detection_alpha_a': ACTIVITY_DETECTION_ALPHA_A,
    'fe.activity_detection_alpha_b': ACTIVITY_DETECTION_ALPHA_B,
    'fe.activity_detection_arm_threshold': ACTIVITY_DETECTION_ARM_THRESHOLD,
    'fe.activity_detection_trip_threshold': ACTIVITY_DETECTION_TRIP_THRESHOLD,
    'fe.dc_notch_filter_enable': DC_NOTCH_FILTER_ENABLE,
    'fe.dc_notch_filter_coefficient': DC_NOTCH_FILTER_COEFFICIENT,
    'fe.quantize_dynamic_scale_enable': QUANTIZE_DYNAMIC_SCALE_ENABLE,
    'fe.quantize_dynamic_scale_range_db': QUANTIZE_DYNAMIC_SCALE_RANGE_DB,
    'fe.fft_length': FFT_LENGTH
}

print(f"✓ Config loaded: SILICON-8 with MLTK AudioFeatureGenerator")
print(f"  - Sample rate: {SAMPLE_RATE_HZ} Hz")
print(f"  - Sample length: {SAMPLE_LENGTH_MS} ms")
print(f"  - Spectrogram shape: {SPECTROGRAM_SHAPE}")
print(f"  - Model parameters budget: {MAX_PARAMS:,}")
