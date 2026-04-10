"""
SILICON-19: COMPACT HARDWARE-AWARE MODEL
- Input: (32, 198, 1) - Int8 Quantized Features
- Optimized for Silicon Labs xG26
"""

from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(32, 198, 1), num_classes=3):
    weight_decay = 1e-4
    reg = regularizers.l2(weight_decay)
    filters = 8

    model = models.Sequential([
        # Layer 1: Conv + BN + Relu
        # Shape: (32, 198, 1) -> (16, 99, 8)
        layers.Conv2D(filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # Layer 2: Conv + BN + Relu + Dropout
        # Shape: (16, 99, 8) -> (8, 50, 16)
        layers.Conv2D(2*filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.1),

        # Layer 3: Mid-level Harmonic Merger (Harmonic modulation capture)
        # Shape: (8, 50, 16) -> (4, 50, 32)
        layers.Conv2D(4*filters, (3,3), strides=(1,1), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2,1)), 

        # Layer 4: Conv + BN + Relu + Dropout
        # Shape: (4, 50, 32) -> (2, 25, 32)
        layers.Conv2D(4*filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # Layer 5: Global Frequency Pooling (Collapse remaining 2 frequency bands)
        # Shape: (2, 25, 32) -> (1, 25, 32)
        layers.MaxPooling2D(pool_size=(2,1)),
        
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()
    print("\n🟢 SILICON-19: Model Architecture Loaded.")
