"""
SILICON-17: MLTK Architecture (Pure Keras)
3-Class Model: Baby Cry, Dog, Unknown
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(64, 198, 1), num_classes=3):
    weight_decay = 1e-4
    reg = regularizers.l2(weight_decay)
    filters = 8

    model = models.Sequential([
        # Layer 1: Conv + BN + Relu
        # Shape: (64, 198, 1) -> (32, 99, 8)
        layers.Conv2D(filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # Layer 2: Conv + BN + Relu + Dropout
        # Shape: (32, 99, 8) -> (16, 50, 16)
        layers.Conv2D(2*filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.1),

        # Layer 3: Conv + BN + Relu + Dropout
        # Shape: (16, 50, 16) -> (8, 25, 32)
        layers.Conv2D(4*filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # Layer 4: Global Frequency Pooling (Collapse the 8 remaining height freq bands)
        # Shape: (8, 25, 32) -> (1, 25, 32)
        layers.MaxPooling2D(pool_size=(8,1)),
        
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()
