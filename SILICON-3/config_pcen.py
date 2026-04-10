import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset_pcen_8s')  # Separate dir for PCEN features 8s
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ==========================================
# EDGE-OPTIMIZED AUDIO PROCESSING
# ==========================================

# Audio parameters
SAMPLE_RATE = 16000       # 16kHz (standard for edge devices)

# STFT parameters
N_FFT = 512               # FFT size
HOP_LENGTH = 160          # 10ms hop for streaming (160 samples @ 16kHz)
                          # Faster than 20ms (320) - better temporal resolution

# Mel filterbank - REDUCED for edge
N_MELS = 32               # Reduced from 64 → Edge-friendly
                          # Still captures perceptual information
                          # Silicon Labs examples use 10-32 bins

# Frequency range
FMIN = 50                 # Capture low-freq thumps/footsteps
FMAX = 8000               # Sufficient for baby cry + dog bark + glass

# ==========================================
# STREAMING WINDOW CONFIGURATION
# ==========================================

# For edge deployment: Process 8-second windows (INCREASED FOR ACCURACY TEST)
WINDOW_DURATION = 8.0     # 8 second windows (was 1.0)
WINDOW_FRAMES = int((WINDOW_DURATION * SAMPLE_RATE) / HOP_LENGTH)
                          # = (8.0 * 16000) / 160 = 800 frames

HOP_DURATION = 0.1        # 100ms hop between windows
HOP_FRAMES = int((HOP_DURATION * SAMPLE_RATE) / HOP_LENGTH)
                          # = (0.1 * 16000) / 160 = 10 frames

# For training/testing: Use full context
MAX_TIME_FRAMES = WINDOW_FRAMES  # 800 frames = 8 seconds (INCREASED from 100)
                                 # Matches baseline for fair comparison

# ==========================================
# PCEN PARAMETERS (NOVEL PREPROCESSING)
# ==========================================

# PCEN replaces log compression with adaptive gain control
# These are standard values from PCEN paper (trainable in advanced setup)

PCEN_ALPHA = 0.98         # Temporal smoothing (0.98 = ~50 frame time constant)
                          # Higher = slower adaptation
PCEN_DELTA = 2.0          # Bias (prevents division by zero)
PCEN_R = 0.5              # Root compression (0.5 = sqrt, smoother than log)
PCEN_S = 0.025            # Gain normalization strength
                          # Higher = stronger normalization

# Optional: Make PCEN trainable
PCEN_TRAINABLE = False    # Set True to learn PCEN params during training
                          # (adds ~4 parameters - negligible)

# ==========================================
# DATA AUGMENTATION - Edge-Aware
# ==========================================

USE_AUGMENTATION = True

# SpecAugment - Conservative (don't destroy edge-critical features)
TIME_MASK_PARAM = 15      # Reduced from 40 (shorter windows)
FREQ_MASK_PARAM = 6       # Reduced from 12 (fewer mel bins)
NUM_MASKS = 1             # Reduced from 2

# MixUp - Keep strong
MIXUP_ALPHA = 0.4         # Same as before

# Noise injection
NOISE_INJECTION_PROB = 0.3
NOISE_STD = 0.005

# ==========================================
# MODEL ARCHITECTURE
# ==========================================

NUM_CLASSES = 2           # Currently: baby_cry, dog
                          # Future: +unusual, +unknown

MAX_PARAMS = 17000        # Silicon Labs edge constraint

# Optimized for smaller input (32×100 vs 64×400)
D_MODEL = 18              # Keep same
NUM_HEADS = 3             # Keep same
NUM_LAYERS = 3            # Keep same
DROPOUT = 0.15            # Keep same

# SE Block
USE_SE_BLOCK = True
SE_REDUCTION = 3

# ==========================================
# TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 250
PATIENCE = 25
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# Warmup + Cosine schedule
USE_WARMUP = True
WARMUP_EPOCHS = 5

# Focal Loss
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

CLASS_NAMES = ['Baby_cry', 'dog']

# ==========================================
# EDGE DEPLOYMENT NOTES
# ==========================================

EDGE_DEPLOYMENT_INFO = """
PCEN vs Log-Mel Advantages for Edge:

1. Robustness:
   - PCEN adapts to volume changes (crying close vs far)
   - Better for quiet sounds (footsteps, soft thumps)
   - Handles mic gain variations (real-world deployment)

2. Computational Efficiency:
   - PCEN uses simple IIR filter (low compute)
   - No expensive logarithm operations
   - Fixed-point friendly (int8 quantization)

3. Perceptual Quality:
   - Automatic gain control mimics human hearing
   - Better dynamic range than log
   - Preserves transients (glass break, bark onset)

4. Streaming-Ready:
   - Temporal IIR naturally handles streaming
   - No need to buffer full clips
   - Constant memory footprint

5. Trainable (optional):
   - Can learn optimal PCEN params per class
   - Only adds ~4 parameters
   - Literature shows this often improves accuracy

Memory Footprint (per window):
- Input: 32 mel bins × 100 frames × 4 bytes = 12.8 KB
- vs Original: 64 bins × 400 frames × 4 bytes = 102.4 KB
- Reduction: 8× smaller!

Inference Latency:
- With 100-frame windows: ~3× faster than 400-frame
- Streaming hop: Classify every 100ms
- Detection delay: <200ms (near real-time)

Novelty Statement:
"We replace standard log-mel spectrograms with PCEN features,
which provide adaptive per-channel gain control and root
compression, improving robustness to volume variations and
better preserving quiet transient events critical for edge
deployment in home monitoring scenarios."
"""

# ==========================================
# FUTURE EXPANSION
# ==========================================

# When adding "unusual" and "unknown" classes:
FUTURE_CLASSES = ['baby_cry', 'dog', 'unusual', 'unknown']
# unusual: glass break, thumps, footsteps
# unknown: adult speech, TV, appliances, ambient

# Streaming detection parameters (for deployment)
STREAMING_CONFIG = {
    'averaging_window_ms': 300,      # Smooth detections over 300ms
    'suppression_ms': 700,           # Suppress repeat notifications
    'minimum_count': 2,              # Require 2+ positives before notifying
    'confidence_threshold': 0.7,     # Minimum confidence
}

# Class-specific thresholds
CLASS_THRESHOLDS = {
    'baby_cry': {
        'threshold': 0.7,
        'min_duration_ms': 500,      # Require 500ms persistence
        'fast_trigger': False,
    },
    'dog': {
        'threshold': 0.7,
        'min_duration_ms': 300,
        'fast_trigger': False,
    },
    'unusual': {  # Future
        'threshold': 0.8,            # Higher threshold (reduce false alarms)
        'min_duration_ms': 100,      # Fast trigger for glass break
        'fast_trigger': True,
    },
    'unknown': {  # Future
        'threshold': 0.5,            # Lower threshold (catch-all)
        'min_duration_ms': 1000,
        'fast_trigger': False,
    },
}
