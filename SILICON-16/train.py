"""
SILICON-16: BINARY TRAINING (Baby Cry vs Not Cry)
- 2.0s context window
- (64, 198) input shape
- Signal-focused architecture
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import create_binary_model

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'unknown']
BATCH_SIZE = 32
EPOCHS = 100
INPUT_SHAPE = (64, 198, 1)

def train():
    print("\n" + "="*50)
    print("      SILICON-16: STRATEGIC BINARY TRAINING")
    print("="*50)
    
    X = []
    y = []

    # 2. Load Pre-processed Files
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            print(f"❌ Error: {class_dir} not found. Run prepare_data.py first!")
            return
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f"📦 Loading {label}: {len(files)} samples")
        
        for f in files:
            feat = np.load(f)
            X.append(feat[..., np.newaxis])
            y.append(idx)

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    
    print(f"\n✅ Total Samples : {len(X)}")
    print(f"📊 Feature Shape : {X.shape[1:]}")

    # 3. Split
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # 4. Model Setup
    model = create_binary_model(input_shape=INPUT_SHAPE, num_classes=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 5. Training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_binary_model.h5', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ]

    print("\n🚀 Training start...")
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
    
    plt.savefig('binary_training_report.png')
    print("\n✅ Training Complete. Best model saved to: best_binary_model.h5")
    print("📈 Training report saved to: binary_training_report.png")

if __name__ == '__main__':
    train()
