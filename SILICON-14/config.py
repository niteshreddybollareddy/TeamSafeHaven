"""
SILICON-14: Judge-Proof Demo Configuration (3-Class Pure)
Source: SILICON-10 exclusively
Pattern: Balanced 1:1:1 Training
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# EXCLUSIVE SOURCE
SOURCE_DIR = os.path.join(BASE_DIR, 'Dataset')

PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ============================================================================
# MLTK AudioFeatureGenerator Settings (JUDGE-PROOF CALIBRATION)
# ============================================================================

SAMPLE_RATE_HZ = 16000
SAMPLE_LENGTH_MS = 1500          # 1.5 seconds target
NR_WARMUP_MS = 500               # 0.5s Silence padding for NR warm-up
TOTAL_INPUT_MS = SAMPLE_LENGTH_MS + NR_WARMUP_MS

# Audio calculation
SAMPLE_LENGTH = int((SAMPLE_LENGTH_MS * SAMPLE_RATE_HZ) / 1000)
WARMUP_SAMPLES = int((NR_WARMUP_MS * SAMPLE_RATE_HZ) / 1000)

# Spectrogram Settings
WINDOW_SIZE_MS = 30              
WINDOW_STEP_MS = 10
FILTERBANK_N_CHANNELS = 64
FILTERBANK_UPPER_BAND_LIMIT = 7500.0
FILTERBANK_LOWER_BAND_LIMIT = 125.0     # Ignore dev-board hum (PRO TIP)

# MLTK Advanced Features (Enabled for Real-time Stability)
NOISE_REDUCTION_ENABLE = True
NOISE_REDUCTION_MIN_SIGNAL_REMAINING = 0.40
DC_NOTCH_FILTER_ENABLE = True
DC_NOTCH_FILTER_COEFFICIENT = 0.95
LOG_SCALE_ENABLE = True
QUANTIZE_DYNAMIC_SCALE_ENABLE = True
QUANTIZE_DYNAMIC_SCALE_RANGE_DB = 40.0

# Shape Calculation
WINDOW_SIZE_LENGTH = int((WINDOW_SIZE_MS * SAMPLE_RATE_HZ) / 1000)
WINDOW_STEP_LENGTH = int((WINDOW_STEP_MS * SAMPLE_RATE_HZ) / 1000)
N_TIME_FRAMES = (SAMPLE_LENGTH - WINDOW_SIZE_LENGTH) // WINDOW_STEP_LENGTH + 1
SPECTROGRAM_SHAPE = (FILTERBANK_N_CHANNELS, N_TIME_FRAMES) # (64, 148)

# ============================================================================
# 3-Class Configuration (Mentor Approved)
# ============================================================================

NUM_CLASSES = 3
# Exact folder names in SILICON-10
CLASS_MAPPING = {
    'Baby Cry': 'baby cry',
    'Dog': 'dog',
    'Unknown': 'unknown'
}
CLASS_NAMES = list(CLASS_MAPPING.keys())
MAX_PARAMS = 15000 

# ============================================================================
# Training Settings
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATIENCE = 15
WEIGHT_DECAY = 1e-4

# Real-time Post-processing
MINIMUM_COUNT = 2                
DETECTION_THRESHOLD = 0.85       

print(f"✓ Config Initialized: SILICON-14 (PURE SILICON-10 SOURCE)")
