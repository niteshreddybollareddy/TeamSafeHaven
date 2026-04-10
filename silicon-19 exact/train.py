"""
SILICON-19: HARDWARE-AWARE TRAINING (16kHz INT8)
- Parity with Silicon Labs xG26 Frontend
- Input: (32, 198, 1) - Int8 Quantized Features
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
INPUT_SHAPE = (32, 198, 1)

def train():
    print("\n" + "="*50)
    print("      SILICON-19: HARDWARE-AWARE 3-CLASS TRAINING")
    print("="*50)
    
    X = []
    y = []

    # 2. Load Pre-processed Files
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: {class_dir} not found. Skipping...")
            continue
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f"📦 Loading {label}: {len(files)} samples")
        
        for f in files:
            feat = np.load(f)
            X.append(feat[..., np.newaxis])
            y.append(idx)

    if not X:
        print("❌ Error: No samples found! Run prepare_data.py first.")
        return

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))
    
    print(f"\n✅ Total Samples : {len(X)}")
    print(f"📊 Feature Shape : {X.shape[1:]}")

    # 3. Split
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # 4. Model Setup
    model = create_model(input_shape=INPUT_SHAPE, num_classes=len(CLASSES))
    
    # Custom Metrics for Baby Cry (Class 0)
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(class_id=0, name='precision_baby'),
        tf.keras.metrics.Recall(class_id=0, name='recall_baby')
    ]
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    # 5. Training
    model_save_path = os.path.join(BASE_DIR, 'best_model.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, 
            save_best_only=True, 
            monitor='val_precision_baby', 
            mode='max',
            verbose = 1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_precision_baby',
            patience=30, 
            mode='max',
            restore_best_weights=True,
            verbose = 1
        )
    ]

    print("\n🚀 Training start (Focusing on Baby Cry Precision)...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # 6. Plot Results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(BASE_DIR, 'training_report.png'))
    
    # Final explicit save (restored best weights)
    model.save(model_save_path)
    
    print(f"\n✅ Training Complete. Best model saved to: {model_save_path}")
    print(f"📈 Training report saved to: {os.path.join(BASE_DIR, 'training_report.png')}")

if __name__ == '__main__':
    train()
