"""
Configuration for Multi-Resolution Audio Classification
Mel + Gammatone + Delta Features for Edge Deployment
"""

import os

# ============================================================
# PATHS
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
PROCESSED_DIR = os.path.join(ROOT_DIR, 'processed_dataset_multiresolution')

# ============================================================
# AUDIO PROCESSING PARAMETERS
# ============================================================
SAMPLE_RATE = 16000         # 16 kHz for edge deployment
N_MELS = 16                 # Mel bins (reduced to make room for gammatone)
N_GAMMATONE = 16            # Gammatone filterbank channels
N_STATIC = N_MELS + N_GAMMATONE  # 32 static features
N_DELTA = N_STATIC          # 32 delta features
N_TOTAL_FEATURES = N_STATIC + N_DELTA  # 64 total channels

HOP_LENGTH = 160            # 10ms hop (160 samples @ 16kHz)
N_FFT = 512                 # FFT size
FMIN = 50                   # Minimum frequency (Hz)
FMAX = 8000                 # Maximum frequency (Hz)

# Temporal parameters
MAX_TIME_FRAMES = 100       # 1.0 second windows (100 frames × 10ms)
WINDOW_DURATION = MAX_TIME_FRAMES * HOP_LENGTH / SAMPLE_RATE  # 1.0 seconds

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
# Input shape: (batch, 1, 64, 100)
#   - 64 channels: [16 Mel + 16 Gamma + 16 Mel-delta + 16 Gamma-delta]
#   - 100 time frames (1 second)

# Transformer parameters (optimized for <17K params)
D_MODEL = 18                # Embedding dimension
NUM_HEADS = 3               # Attention heads (18 / 3 = 6 per head)
NUM_LAYERS = 3              # Transformer layers
DROPOUT = 0.15              # Dropout rate

# CNN parameters
USE_SE_BLOCK = True         # Squeeze-and-Excitation blocks
SE_REDUCTION = 3            # SE reduction ratio

# Parameter budget
MAX_PARAMS = 17000          # Silicon Labs constraint

# ============================================================
# TRAINING PARAMETERS
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 3e-4        # 0.0003
WEIGHT_DECAY = 1e-5
EPOCHS = 250
EARLY_STOPPING_PATIENCE = 25

# Learning rate schedule
USE_WARMUP = True
WARMUP_EPOCHS = 5
LR_SCHEDULE = 'cosine'      # 'cosine' or 'step'

# Loss function
LOSS_TYPE = 'focal'         # 'focal' or 'crossentropy'
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Gradient clipping
GRAD_CLIP = 1.0

# ============================================================
# DATA AUGMENTATION
# ============================================================
USE_AUGMENTATION = True

# SpecAugment parameters (adjusted for 64 channels, 100 frames)
TIME_MASK_PARAM = 15        # Max time mask width (1 second / 100 frames)
FREQ_MASK_PARAM = 12        # Max freq mask width (64 channels)
NUM_MASKS = 1               # Number of masks per type

# MixUp augmentation
USE_MIXUP = True
MIXUP_ALPHA = 0.4

# ============================================================
# DATASET SPLIT
# ============================================================
TRAIN_SPLIT = 0.70          # 70% training
VAL_SPLIT = 0.15            # 15% validation
TEST_SPLIT = 0.15           # 15% test

# ============================================================
# CLASS CONFIGURATION
# ============================================================
CLASS_NAMES = ['Baby_cry', 'dog']
NUM_CLASSES = len(CLASS_NAMES)

# Future expansion for "unusual" detection
# FUTURE_CLASSES = ['Baby_cry', 'dog', 'unusual', 'unknown']
# where unusual = glass break, thumps, etc.

# ============================================================
# HARDWARE DEPLOYMENT (Silicon Labs xG26)
# ============================================================
TARGET_DEVICE = 'xG26'
TARGET_RAM = 512 * 1024     # 512 KB RAM
TARGET_FLASH = 3200 * 1024  # 3.2 MB Flash
TARGET_CPU_MHZ = 78         # 78 MHz Cortex-M33

# Edge inference constraints
MAX_INFERENCE_TIME_MS = 200 # Max latency (ms)
STREAMING_WINDOW_MS = 1000  # 1 second sliding windows
STREAMING_HOP_MS = 100      # 100ms hop for streaming

# Model quantization (future)
QUANTIZE_MODEL = False      # 8-bit quantization for deployment
QUANTIZATION_BITS = 8       # INT8 quantization

# ============================================================
# FEATURE MEMORY FOOTPRINT
# ============================================================
# Single sample memory:
#   64 channels × 100 frames × 4 bytes (float32) = 25,600 bytes = 25.6 KB
# Batch memory (32 samples):
#   32 × 25.6 KB = 819.2 KB (well within 512 KB with streaming)

FEATURE_MEMORY_PER_SAMPLE = N_TOTAL_FEATURES * MAX_TIME_FRAMES * 4  # bytes
BATCH_MEMORY = FEATURE_MEMORY_PER_SAMPLE * BATCH_SIZE  # bytes

# ============================================================
# LOGGING & CHECKPOINTING
# ============================================================
CHECKPOINT_DIR = 'models'
LOG_DIR = 'logs'
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_f1'   # Monitor F1 score for best model

# Visualization
PLOT_TRAINING_CURVES = True
SAVE_CONFUSION_MATRIX = True

# ============================================================
# FEATURE INTERPRETATION
# ============================================================
FEATURE_GROUPS = {
    'mel_static': (0, 16),        # Indices [0:16]
    'gamma_static': (16, 32),     # Indices [16:32]
    'mel_delta': (32, 48),        # Indices [32:48]
    'gamma_delta': (48, 64)       # Indices [48:64]
}

# ============================================================
# PRINT CONFIGURATION SUMMARY
# ============================================================
def print_config():
    print("=" * 60)
    print("MULTI-RESOLUTION CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {PROCESSED_DIR}")
    print(f"\nFeature Configuration:")
    print(f"  Mel bins: {N_MELS}")
    print(f"  Gammatone bins: {N_GAMMATONE}")
    print(f"  Delta features: {N_DELTA}")
    print(f"  Total channels: {N_TOTAL_FEATURES}")
    print(f"  Time frames: {MAX_TIME_FRAMES} ({WINDOW_DURATION:.1f}s)")
    print(f"  Input shape: (batch, 1, {N_TOTAL_FEATURES}, {MAX_TIME_FRAMES})")
    print(f"\nModel Architecture:")
    print(f"  d_model: {D_MODEL}")
    print(f"  num_heads: {NUM_HEADS}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  dropout: {DROPOUT}")
    print(f"  SE blocks: {USE_SE_BLOCK}")
    print(f"  Parameter budget: {MAX_PARAMS:,}")
    print(f"\nTraining:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Loss: {LOSS_TYPE}")
    print(f"  Augmentation: {USE_AUGMENTATION}")
    print(f"\nMemory Footprint:")
    print(f"  Per sample: {FEATURE_MEMORY_PER_SAMPLE / 1024:.1f} KB")
    print(f"  Batch ({BATCH_SIZE}): {BATCH_MEMORY / 1024:.1f} KB")
    print(f"  Target RAM: {TARGET_RAM / 1024:.0f} KB")
    print("=" * 60)

if __name__ == '__main__':
    print_config()
