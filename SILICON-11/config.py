"""
SILICON-11 Configuration
Lightweight 2D CNN for 4-Class Audio Classification
Using variable-length audio support (same as SILICON-7)
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ============================================================================
# Audio Processing Settings (same as SILICON-7)
# ============================================================================

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 320
MAX_TIME_FRAMES = 500  # ~10s max buffer for variable-length training

# SILICON-11 natural durations (ALL KEPT AS-IS, NO SEGMENTATION):
#   - Baby Cry: ~7s (natural recordings with 1-2s cry bursts inside)
#   - Dog: 0.13s-20s (natural barks, outliers >20s removed)
#   - Unusual: 7-8s (natural sounds)
#   - Unknown: 5-20s (natural background, outliers >20s removed)
# Variable-length batching handles all durations efficiently

# ============================================================================
# Model Architecture Settings - LIGHTWEIGHT CNN
# ============================================================================

# 4-class classification: Baby Cry, Dog, Unknown, Unusual
NUM_CLASSES = 4
MAX_PARAMS = 10000

# Lightweight 2D CNN configuration
MODEL_TYPE = 'cnn'  # Use lightweight CNN (not transformer)
USE_DEPTHWISE_SEPARABLE = True  # Reduces params by ~6x
CNN_FILTERS = [8, 16, 24, 32]  # Progressive channel growth
CNN_DROPOUT = 0.2

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

# Class names (4 classes)
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
    "Lightweight 2D CNN (depthwise separable convolutions)",
    "Variable-length audio support (0.13s to 20s)",
    "No padding during preprocessing (natural durations preserved)",
    "GlobalAveragePooling for variable-length handling",
    "Focal Loss for class imbalance",
    "Warmup + Cosine LR schedule for stable training",
]

EXPECTED_PERFORMANCE = {
    'accuracy': '88-92%',
    'inference_time_ms': '<20',
    'model_size_kb': '<10',
    'parameters': '<10000'
}

print(f"✓ Config loaded: SILICON-11 Lightweight CNN")
print(f"  - Sample rate: {SAMPLE_RATE} Hz")
print(f"  - Mel bins: {N_MELS}")
print(f"  - Model type: {MODEL_TYPE}")
print(f"  - Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
print(f"  - Model parameters budget: {MAX_PARAMS:,}")
