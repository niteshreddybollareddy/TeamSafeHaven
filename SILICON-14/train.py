"""
SILICON-14: Real-time Optimized Training
Model: TENet-Micro
Strategy: Balanced Epochs + Demo-Confidence Scaling
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import config
from model import create_model
from preprocess import ProPreprocessor

def plot_training_results(history, output_path):
    """Save training progress graphs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(output_path)
    print(f"📊 Training graphs saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title('Confusion Matrix - SILICON-14')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    print(f"📉 Confusion matrix saved to {output_path}")

def load_processed_data():
    X = []
    y = []
    
    # First, count files in each class to find the minimum
    class_counts = {}
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        if os.path.exists(class_dir):
            class_counts[class_name] = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
    
    # FIND THE MINIMUM SIZE (likely 999)
    min_samples = min(class_counts.values()) if class_counts else 0
    
    print(f"\n📦 Loading Balanced Dataset (Cap: {min_samples} samples per class)...")
    prep = ProPreprocessor() # NEW: Uses stats.json automatically
    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        # Shuffle and Cap
        np.random.seed(42)
        np.random.shuffle(files)
        files = files[:min_samples]
        
        print(f"  - {class_name:10}: Loaded {len(files)} samples")
        
        for f in files:
            feat = np.load(f)
            feat = feat[..., np.newaxis]
            X.append(feat)
            y.append(idx)
            
    return np.array(X), np.array(y)

import sys

class DualLogger(object):
    """Logs terminal output to both console and a file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def train():
    log_path = os.path.join(config.LOG_DIR, 'training_log.txt')
    sys.stdout = DualLogger(log_path)
    
    print("\n" + "="*60)
    print("      SILICON-14: EXHAUSTIVE TRAINING SYSTEM")
    print(f"      Log File: {log_path}")
    print("="*60)
    
    # 1. Load and Verify
    X, y = load_processed_data()
    if len(X) == 0:
        print("❌ CRITICAL: No data loaded. Training aborted.")
        return

    # One-hot encoding
    y_cat = tf.keras.utils.to_categorical(y, num_classes=config.NUM_CLASSES)
    
    # 2. Split (with stratify to maintain balance)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"\n✅ Data Split Complete:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Shape: {X_train[0].shape}")

    # 3. Initialize Model
    print("\n🏗️ Building TENet-Micro (Hardware Optimized for xG26)...")
    model = create_model(num_classes=config.NUM_CLASSES)
    
    # Schedule
    lr_schedule = optimizers.schedules.ExponentialDecay(
        config.LEARNING_RATE,
        decay_steps=500,
        decay_rate=0.96,
        staircase=True
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # 4. Callbacks
    h5_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            h5_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=config.PATIENCE, 
            monitor='val_accuracy',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Progress Logger - UPDATED WITH ALL 4 METRICS
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"   [Epoch {epoch+1:3}] "
                f"Train_Acc: {logs['accuracy']:.4f} | Train_Loss: {logs['loss']:.4f} | "
                f"Val_Acc: {logs['val_accuracy']:.4f} | Val_Loss: {logs['val_loss']:.4f}"
            )
        )
    ]
    
    # 5. Execute Training
    print("\n🚀 Training in progress... (Monitoring Val_Accuracy for Overfitting)")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=0 # Using LambdaCallback for cleaner terminal output
    )
    
    # 6. Post-Training Analysis
    print("\n" + "="*60)
    print("      POST-TRAINING ANALYTICS REPORT")
    print("="*60)
    
    # Save Graphs
    plot_training_results(history, os.path.join(config.LOG_DIR, 'training_history.png'))
    
    # Generate Confusion Matrix on Validation Set
    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    
    plot_confusion_matrix(
        y_val_true, 
        y_val_pred, 
        config.CLASS_NAMES, 
        os.path.join(config.LOG_DIR, 'confusion_matrix.png')
    )
    
    # Classification Report
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_val_true, y_val_pred, target_names=config.CLASS_NAMES))
    
    print(f"\n🏆 Best Target: Val Accuracy > 90%")
    final_acc = max(history.history['val_accuracy'])
    if final_acc >= 0.90:
        print(f"   RESULTS: SUCCESS! ({final_acc:.2%})")
    else:
        print(f"   RESULTS: NEEDS IMPROVEMENT ({final_acc:.2%})")
        
    print(f"\n💾 Model Weights: {h5_path}")
    print("="*60 + "\n")

if __name__ == '__main__':
    if not os.path.exists(config.PROCESSED_DIR) or len(os.listdir(config.PROCESSED_DIR)) == 0:
        print("⚠ processed_dataset is empty. Please run python prepare_data.py first!")
    else:
        train()
