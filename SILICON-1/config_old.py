import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Audio processing
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
MAX_TIME_FRAMES = 437  # ~7 seconds at 16kHz with hop_length=256

# Data augmentation
USE_AUGMENTATION = True
TIME_MASK_PARAM = 40
FREQ_MASK_PARAM = 8
NUM_MASKS = 2

# Model
NUM_CLASSES = 2
MAX_PARAMS = 17000

# Training - UPDATED
BATCH_SIZE = 32  # Reduced from 64 for better gradient estimates
LEARNING_RATE = 3e-4  # Reduced from 1e-3 for more stable training
NUM_EPOCHS = 100  # Increased from 50
PATIENCE = 15  # Increased early stopping patience
WEIGHT_DECAY = 1e-5  # Reduced regularization
GRAD_CLIP = 1.0  # Add gradient clipping

# Learning rate scheduling
USE_COSINE_SCHEDULE = True  # Alternative to ReduceLROnPlateau
WARMUP_EPOCHS = 5

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

CLASS_NAMES = ['cry', 'not_cry']

# Class weights for imbalanced data
USE_CLASS_WEIGHTS = True  # Enable if dataset is imbalanced