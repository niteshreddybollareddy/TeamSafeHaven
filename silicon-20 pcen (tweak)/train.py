"""
SILICON-20: 3-CLASS TRAINING (Baby Cry, Dog, Unknown)
- Optimized for PCEN Features (64, 198)
- Stratified sampling and z-score normalization
- Performance focus on Baby Cry precision/recall
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import create_model

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'dog', 'unknown']
BATCH_SIZE = 32
EPOCHS = 300
INPUT_SHAPE = (64, 198, 1)

def train():
    print("\n" + "="*60)
    print("      SILICON-20: PCEN-BASED 3-CLASS TRAINING (V1)")
    print("="*60)
    
    X = []
    y = []

    # 2. Load Pre-processed PCEN Files
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found. Skipping...")
            continue
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f"Loading {label}: {len(files)} samples")
        
        for f in files:
            feat = np.load(f)
            # Ensure shape (64, 198, 1)
            if feat.shape != INPUT_SHAPE[:2]:
                if feat.shape[0] == INPUT_SHAPE[0]:
                    # Likely (64, T)
                    if feat.shape[1] > INPUT_SHAPE[1]:
                        feat = feat[:, :INPUT_SHAPE[1]]
                    else:
                        feat = np.pad(feat, ((0,0), (0, INPUT_SHAPE[1] - feat.shape[1])), mode='constant')
                else:
                    continue # Skip invalid shapes
            
            X.append(feat[..., np.newaxis])
            y.append(idx)

    if not X:
        print("Error: No samples found! Run prepare_data.py first.")
        return

    X = np.array(X, dtype=np.float32)
    labels_raw = np.array(y) # For stratification
    y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))
    
    # 3. Global Feature Normalization (Z-Score)
    # Crucial for PCEN range stability
    print("\nNormalizing PCEN features...")
    mean = X.mean(axis=(0,1,2), keepdims=True)
    std = X.std(axis=(0,1,2), keepdims=True) + 1e-9
    X = (X - mean) / std
    
    # Save normalization stats for inference consistency
    np.save(os.path.join(BASE_DIR, 'normalization_stats.npy'), {'mean': mean, 'std': std})
    print(f"Saved normalization stats to: normalization_stats.npy")

    print(f"Total Samples : {len(X)}")
    print(f"Feature Shape : {X.shape[1:]}")

    # 4. Stratified Split
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.15, 
        random_state=42, 
        stratify=labels_raw
    )

    # 5. Model Setup
    model = create_model(input_shape=INPUT_SHAPE, num_classes=len(CLASSES))
    
    # Comprehensive Metrics for Debugging
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(class_id=0, name='precision_baby'),
        tf.keras.metrics.Recall(class_id=0, name='recall_baby'),
        tf.keras.metrics.Precision(class_id=1, name='precision_dog'),
        tf.keras.metrics.Recall(class_id=1, name='recall_dog'),
        tf.keras.metrics.Precision(class_id=2, name='precision_unknown'),
        tf.keras.metrics.Recall(class_id=2, name='recall_unknown')
    ]
    
    # Lower initial LR for PCEN stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    # 6. Training with PCEN-optimized callbacks and class weighting
    # Boost baby cry (0) weight to ensure high sensitivity
    class_weights = {0: 1.2, 1: 1.0, 2: 1.0}
    
    model_save_path = os.path.join(BASE_DIR, 'best_model.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, 
            save_best_only=True, 
            monitor='val_precision_baby', 
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_precision_baby',
            patience=40, 
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1
        )
    ]

    print("\nStarting Training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=True,
        verbose=1
    )


    plt.figure(figsize=(15, 5))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Global Accuracy')
    plt.legend()
    
    # Baby Precision/Recall
    plt.subplot(1, 3, 2)
    plt.plot(history.history['precision_baby'], label='Precision')
    plt.plot(history.history['recall_baby'], label='Recall')
    plt.title('Baby Cry Metrics (Val)')
    plt.legend()
    
    # Loss
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.savefig('training_report.png')
    
    # Final save of the restored best weights
    model.save(model_save_path)
    
    print(f"\nTraining Complete. PCEN-stable model saved to: {model_save_path}")
    print("Detailed training metrics saved to: training_report.png")

if __name__ == '__main__':
    train()
