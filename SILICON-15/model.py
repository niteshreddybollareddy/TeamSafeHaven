"""
SILICON-15: MLTK Architecture (Pure Keras)
This is an exact replication of the Silicon Labs CNN used in their examples.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_binary_model(input_shape=(64, 148, 1), num_classes=2):
    weight_decay = 1e-4
    reg = regularizers.l2(weight_decay)
    filters = 8

    model = models.Sequential([
        # Layer 1: Conv + BN + Relu
        layers.Conv2D(filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # Layer 2: Conv + BN + Relu + Dropout
        layers.Conv2D(2*filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.1),

        # Layer 3: Conv + BN + Relu + Dropout
        layers.Conv2D(4*filters, (3,3), strides=(2,2), padding='same', 
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # Layer 4: Global Max Pooling (Fixed window 7x1 based on MLTK)
        layers.MaxPooling2D(pool_size=(7,1)),
        
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    model = create_binary_model()
    model.summary()
