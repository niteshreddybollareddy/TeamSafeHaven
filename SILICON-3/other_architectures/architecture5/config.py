import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_DIR = os.path.join(ROOT_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(ROOT_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Audio processing - OPTIMIZED FOR ACTUAL DATA
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 320
MAX_TIME_FRAMES = 400  # 8.0s coverage (was 128=2.56s)
# Dataset analysis shows: Baby avg 9.07s, Dog avg 7.90s
# 400 frames = 8.0s covers 88% of baby cries, 101% of dog sounds
# Captures full temporal evolution of cry patterns
# NO parameter increase (transformers are sequence-length independent)

# Data augmentation - ENHANCED
USE_AUGMENTATION = True
TIME_MASK_PARAM = 40        # Increased from 30 for more aggressive masking
FREQ_MASK_PARAM = 12        # Increased from 8
NUM_MASKS = 2

# MixUp augmentation - ENHANCED
MIXUP_ALPHA = 0.4           # Increased from 0.2 for more diverse mixing

# Model architecture - IMPROVED
NUM_CLASSES = 2
MAX_PARAMS = 17000

# Improved Transformer architecture (optimized to fit under 17K params)
D_MODEL = 18                # Keep at 18 (3 heads × 6 dims)
NUM_HEADS = 3               # 3 heads for balanced attention
NUM_LAYERS = 3              # Increased from 2 for better capacity
DROPOUT = 0.15              # Slightly increased from 0.1

# Squeeze-and-Excitation block - NEW
USE_SE_BLOCK = True
SE_REDUCTION = 3            # Reduction ratio for SE block (6 channels / 3 = 2)

# Multi-scale CNN configuration - NEW
# Using 3x3 (3 filters) + 5x5 (2 filters) + 7x7 (1 filter) = 6 total
# This configuration fits within 17K parameter budget (15,531 params used)
# Provides multi-scale temporal pattern extraction

# Training - IMPROVED
BATCH_SIZE = 32
LEARNING_RATE = 3e-4        # Peak learning rate after warmup
NUM_EPOCHS = 250            # Extended from 200 for better convergence
PATIENCE = 25               # Increased from 20
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# Learning rate scheduling - IMPROVED WITH WARMUP
USE_WARMUP = True
WARMUP_EPOCHS = 5           # Linear warmup for first 5 epochs
WARMUP_START_LR = 3e-5      # Starting LR for warmup (1/10 of peak)

# After warmup, use cosine decay for remaining epochs
USE_COSINE_SCHEDULE = True

# Focal Loss parameters - NEW (replacing standard cross-entropy)
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0           # Focusing parameter (standard value)
FOCAL_ALPHA = 0.25          # Balancing parameter for classes

# Data split (unchanged - maintain consistency)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class weights (kept for compatibility, but using Focal Loss instead)
USE_CLASS_WEIGHTS = False   # Disabled since Focal Loss handles this

CLASS_NAMES = ['Baby_cry', 'dog']

# Reproducibility
RANDOM_SEED = 42

# Production notes
PRODUCTION_IMPROVEMENTS = [
    "Multi-scale CNN (3x3 + 5x5 + 7x7 kernels) for diverse temporal patterns",
    "Squeeze-and-Excitation blocks for channel attention",
    "Deeper Transformer (3 layers) for better temporal modeling",
    "Warmup + Cosine LR schedule for stable training",
    "Focal Loss for better handling of hard examples",
    "Enhanced augmentation (stronger SpecAugment + MixUp)"
]

# Expected improvements over baseline:
# - Multi-scale CNN: +1.0-1.5%
# - SE blocks: +0.3-0.5%
# - Deeper transformer: +0.5-1.0%
# - Focal Loss: +0.3-0.5%
# - Warmup schedule: +0.2-0.3%
# - Enhanced augmentation: +0.5-0.8%
# Total expected: +3.0-5.0% accuracy improvement
