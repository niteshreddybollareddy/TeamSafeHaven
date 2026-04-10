import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_DIR = os.path.join(ROOT_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(ROOT_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Audio processing
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 320
MAX_TIME_FRAMES = 128  # Fixed for this architecture

# Data augmentation
USE_AUGMENTATION = True
TIME_MASK_PARAM = 30
FREQ_MASK_PARAM = 8
NUM_MASKS = 2

# Model
NUM_CLASSES = 2
MAX_PARAMS = 17000  # Parameter budget

# Transformer architecture - optimized to use ~15K of 17K budget
D_MODEL = 18  # Reduced for CNN+Transformer hybrid
NUM_HEADS = 3  # 3 heads with 6 dims each (18/3=6)
NUM_LAYERS = 2
DROPOUT = 0.1

# Training - UPDATED
BATCH_SIZE = 32  # Reduced for better gradient estimates
LEARNING_RATE = 3e-4  # Lower learning rate for transformers
NUM_EPOCHS = 200
PATIENCE = 20  # Increased patience
WEIGHT_DECAY = 1e-5  # Reduced weight decay
GRAD_CLIP = 1.0  # Gradient clipping

# Learning rate scheduling
USE_COSINE_SCHEDULE = True
LR_FACTOR = 0.5
LR_PATIENCE = 6

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class weights for imbalanced data
USE_CLASS_WEIGHTS = True

CLASS_NAMES = ['Baby_cry', 'dog']