"""
SILICON-21: CryNet Training (V1)
- Implements Multi-Task Learning with Auxiliary Losses.
- Forces semantic disentanglement: Envelope, Spectrum, and Rhythm.
- Optimized for PCEN features with Z-score normalization.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import create_model

import sys

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'dog', 'unknown']
BATCH_SIZE = 32
EPOCHS = 350 # Slightly more epochs for multi-task stability
INPUT_SHAPE = (64, 198, 1)

class Logger(object):
    """Dual-stream logger: prints to console and writes to file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def train():
    # Initialize logging to file
    log_file = os.path.join(BASE_DIR, 'training_log.txt')
    sys.stdout = Logger(log_file)
    
    print("\n" + "="*70)
    print("      SILICON-21: CRYNET MULTI-TASK TRAINING (PCEN)")
    print("="*70)
    
    X = []
    y_main = []

    # 2. Load Pre-processed PCEN Files
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: {class_dir} not found. Skipping...")
            continue
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f"📦 Loading {label}: {len(files)} samples")
        
        for f in files:
            feat = np.load(f)
            # Ensure shape (64, 198, 1)
            if feat.shape != INPUT_SHAPE[:2]:
                if feat.shape[0] == INPUT_SHAPE[0]:
                    if feat.shape[1] > INPUT_SHAPE[1]:
                        feat = feat[:, :INPUT_SHAPE[1]]
                    else:
                        feat = np.pad(feat, ((0,0), (0, INPUT_SHAPE[1] - feat.shape[1])), mode='constant')
                else:
                    continue
            
            X.append(feat[..., np.newaxis])
            y_main.append(idx)

    if not X:
        print("❌ Error: No samples found! Run prepare_data.py first.")
        return

    X = np.array(X, dtype=np.float32)
    y_main_raw = np.array(y_main)
    
    # Generate Auxiliary Targets based on CryNet Philosophy
    # 0: baby, 1: dog, 2: unknown
    y_env = []   # Binary: sustained energy?
    y_spec = []  # Categories: cry-like, bark-like, noise
    y_temp = []  # Rhythm score
    
    for label_idx in y_main_raw:
        if label_idx == 0: # Baby
            y_env.append(1.0)
            y_spec.append([1.0, 0.0, 0.0]) # Cry-like
            y_temp.append(1.0)
        elif label_idx == 1: # Dog
            y_env.append(0.0)
            y_spec.append([0.0, 1.0, 0.0]) # Bark-like
            y_temp.append(0.0)
        else: # Unknown
            y_env.append(0.0)
            y_spec.append([0.0, 0.0, 1.0]) # Noise
            y_temp.append(0.0)

    y_main_cat = tf.keras.utils.to_categorical(y_main, num_classes=len(CLASSES))
    y_env = np.array(y_env, dtype=np.float32)
    y_spec = np.array(y_spec, dtype=np.float32)
    y_temp = np.array(y_temp, dtype=np.float32)
    
    # 3. Z-Score Normalization
    print("\n⚖️ Normalizing PCEN features...")
    mean = X.mean(axis=(0,1,2), keepdims=True)
    std = X.std(axis=(0,1,2), keepdims=True) + 1e-9
    X = (X - mean) / std
    np.save(os.path.join(BASE_DIR, 'normalization_stats.npy'), {'mean': mean, 'std': std})

    # Data Splitting (Stratified)
    indices = np.arange(len(X))
    idx_train, idx_val = train_test_split(indices, test_size=0.15, stratify=y_main_raw, random_state=42)
    
    x_train, x_val = X[idx_train], X[idx_val]
    ys_train = {
        "main_output": y_main_cat[idx_train],
        "envelope_aux": y_env[idx_train],
        "spectral_aux": y_spec[idx_train],
        "temporal_aux": y_temp[idx_train]
    }
    ys_val = {
        "main_output": y_main_cat[idx_val],
        "envelope_aux": y_env[idx_val],
        "spectral_aux": y_spec[idx_val],
        "temporal_aux": y_temp[idx_val]
    }

    # 4. Model Setup (Training Mode)
    model = create_model(input_shape=INPUT_SHAPE, num_classes=len(CLASSES), training=True)
    
    # Loss Configuration: Explicit loss objects per output
    # Refined: We rely on the Multi-Pillar architecture and PCEN for bias
    losses = {
        "main_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
        "envelope_aux": tf.keras.losses.BinaryCrossentropy(),
        "spectral_aux": tf.keras.losses.CategoricalCrossentropy(),
        "temporal_aux": tf.keras.losses.MeanSquaredError()
    }
    
    # Loss-level weighting matches the "GoogLeNet-style" auxiliary head philosophy
    loss_weights = {
        "main_output": 1.0,     # Primary classifier dominates
        "envelope_aux": 0.5,    # Envelope is second most important
        "temporal_aux": 0.3,    # Rhythm helps dog rejection
        "spectral_aux": 0.1     # Spectral has lowest priority
    }
    
    metrics = {"main_output": ['accuracy', tf.keras.metrics.Recall(class_id=0, name='recall_baby')]}
    
    # LR optimization for multi-task convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    # 5. Training
    model_save_path = os.path.join(BASE_DIR, 'best_model.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_main_output_accuracy', mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_main_output_loss', patience=50, restore_best_weights=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.5, patience=15, verbose=1, mode='min')
    ]

    print("\n🚀 Starting CryNet Multi-Task Training...")
    # Clean Fit: No hacks, just data + architectural bias
    # verbose=2 prevents the "multiple loading bars" issue for multi-task models
    history = model.fit(
        x_train, ys_train,
        validation_data=(x_val, ys_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2
    )

    # 7. Visualization
    plt.figure(figsize=(18, 5))
    
    # Panel 1: Main Classification Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['main_output_accuracy'], label='Train Acc')
    plt.plot(history.history['val_main_output_accuracy'], label='Val Acc')
    plt.title('Main Output Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 2: Baby Cry Recall (Critical Metric)
    plt.subplot(1, 3, 2)
    plt.plot(history.history['main_output_recall_baby'], label='Train Recall')
    plt.plot(history.history['val_main_output_recall_baby'], label='Val Recall')
    plt.title('Baby Cry Sensitivity (Recall)')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Multi-Task Loss Breakdown
    plt.subplot(1, 3, 3)
    plt.plot(history.history['main_output_loss'], label='Main Loss', alpha=0.8)
    plt.plot(history.history['envelope_aux_loss'], label='Env Loss', linestyle='--', alpha=0.6)
    plt.plot(history.history['temporal_aux_loss'], label='Temp Loss', linestyle='--', alpha=0.6)
    plt.plot(history.history['spectral_aux_loss'], label='Spec Loss', linestyle='--', alpha=0.6)
    plt.title('Loss Breakdown (Pillars)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log') # Log scale helps see small aux losses
    plt.grid(True, alpha=0.3)
    
    report_path = os.path.join(BASE_DIR, 'training_report.png')
    plt.tight_layout()
    plt.savefig(report_path)
    
    print(f"\n✅ Training Complete. CryNet model (Inference-only) saved to: {model_save_path}")
    print(f"📈 Detailed training metrics saved to: {report_path}")

if __name__ == '__main__':
    train()
