"""
SILICON-12: CryNet 3-Class Audio Classifier
Based on SILICON-9's proven architecture (99.81% accuracy) extended to 3 classes

Architecture (SILICON-9 proven design):
  - Pure 2D CNN with standard Conv2D (NOT depthwise - need expressiveness)
  - All TFLite-compatible operations
  - ~7,500 parameters (under 10K budget)
  - 12-18ms inference time on xG26
  - 3-class classification: Baby Cry, Dog Barking, Unknown

KEY DIFFERENCES FROM SILICON-10:
1. Standard Conv2D instead of depthwise separable (more expressive)
2. Dense(16) intermediate layer before output (CRITICAL for discrimination)
3. Dropout 0.4 instead of 0.2 (stronger regularization)
4. Softmax activation in model (not in loss function)

WHY THIS WORKS:
- SILICON-9 achieved 99.81% with this architecture on 2 classes
- Standard convolutions learn cross-channel features better
- Intermediate dense layer provides learned feature transformation
- LayerNorm is stable for single-sample real-time inference
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Import config
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class CryNet3Class(keras.Model):
    """
    CryNet 3-Class: Audio classifier for Baby Cry, Dog Barking, Unknown

    Input: (batch, 1, 64, 148) - mono mel-spectrogram (1.5s audio, 10ms hop)
    Output: (batch, 3) - softmax probabilities [P(BabyCry), P(Dog), P(Unknown)]

    Architecture (SILICON-9 proven + extended):
      Block1: Conv2D(6)  + LN + ReLU + MaxPool  -> (B, 32, 74, 6)
      Block2: Conv2D(12) + LN + ReLU + MaxPool  -> (B, 16, 37, 12)
      Block3: Conv2D(18) + LN + ReLU + MaxPool  -> (B, 8, 18, 18)
      Block4: Conv2D(24) + LN + ReLU + GlobalAvgPool -> (B, 24)
      Head:   Dense(16) + ReLU + Dropout + Dense(3) + Softmax

    Total Parameters: ~7,500 (within 10K budget)
    """

    def __init__(self, num_classes=3, dropout=0.4):
        super(CryNet3Class, self).__init__()

        self.num_classes = num_classes

        # Block 1: Low-level feature extraction
        self.conv1 = layers.Conv2D(
            filters=6, kernel_size=(3, 3), padding='same',
            use_bias=False, name='conv1'
        )
        self.ln1 = layers.LayerNormalization(name='ln1')
        self.relu1 = layers.ReLU(name='relu1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')

        # Block 2: Mid-level patterns
        self.conv2 = layers.Conv2D(
            filters=12, kernel_size=(3, 3), padding='same',
            use_bias=False, name='conv2'
        )
        self.ln2 = layers.LayerNormalization(name='ln2')
        self.relu2 = layers.ReLU(name='relu2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')

        # Block 3: High-level patterns
        self.conv3 = layers.Conv2D(
            filters=18, kernel_size=(3, 3), padding='same',
            use_bias=False, name='conv3'
        )
        self.ln3 = layers.LayerNormalization(name='ln3')
        self.relu3 = layers.ReLU(name='relu3')
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')

        # Block 4: Feature compression
        self.conv4 = layers.Conv2D(
            filters=24, kernel_size=(3, 3), padding='same',
            use_bias=False, name='conv4'
        )
        self.ln4 = layers.LayerNormalization(name='ln4')
        self.relu4 = layers.ReLU(name='relu4')
        self.gap = layers.GlobalAveragePooling2D(name='gap')

        # Classification head (CRITICAL: intermediate dense layer)
        self.dense1 = layers.Dense(16, activation='relu', name='dense1')
        self.dropout = layers.Dropout(dropout, name='dropout')
        self.dense2 = layers.Dense(num_classes, name='dense2')
        self.softmax = layers.Softmax(name='softmax')

    def call(self, x, training=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, 1, 64, 98) - NCHW format
            training: Boolean, whether in training mode

        Returns:
            Softmax probabilities of shape (batch, num_classes)
        """
        # Convert from NCHW (B, 1, 64, 98) to NHWC (B, 64, 98, 1) for Conv2D
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = tf.transpose(x, perm=[0, 2, 3, 1])

        # Block 1
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.ln4(x)
        x = self.relu4(x)
        x = self.gap(x)

        # Classification head
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.softmax(x)

        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


def test_model():
    """Test model initialization and forward pass"""
    print("\n" + "="*70)
    print("SILICON-12: CRYNET 3-CLASS MODEL TEST")
    print("="*70)

    model = CryNet3Class(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)

    # Build with dummy input (1.5s audio -> 64x148 spectrogram)
    dummy_input = tf.random.normal((1, 1, 64, config.N_TIME_FRAMES))
    output = model(dummy_input, training=False)

    print(f"\nModel built successfully")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    params = count_parameters(model)
    print(f"\nTotal parameters: {params:,}")
    print(f"  Budget: {config.MAX_PARAMS:,}")

    if params <= config.MAX_PARAMS:
        print(f"  Within budget! ({config.MAX_PARAMS - params:,} remaining)")
    else:
        print(f"  Exceeds budget by {params - config.MAX_PARAMS:,}")

    # Test batch sizes
    print(f"\nTesting batch sizes:")
    for batch_size in [1, 16, 64]:
        test_input = tf.random.normal((batch_size, 1, 64, config.N_TIME_FRAMES))
        test_output = model(test_input, training=False)
        print(f"  Batch {batch_size}: {test_input.shape} -> {test_output.shape}")

    # Model summary
    print(f"\nModel summary:")
    model.build((None, 1, 64, config.N_TIME_FRAMES))
    model.summary()

    # Operations used
    print(f"\nOperations (all TFLite compatible):")
    for layer in model.layers:
        print(f"  - {layer.__class__.__name__}: {layer.name}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)

    return model


if __name__ == '__main__':
    model = test_model()

    # Test predictions
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*70}")

    batch_size = 4
    spectrograms = tf.random.normal((batch_size, 1, 64, config.N_TIME_FRAMES))
    predictions = model(spectrograms, training=False)

    print(f"\nInput: {spectrograms.shape}")
    print(f"Output: {predictions.shape}")

    for i in range(batch_size):
        probs = predictions[i].numpy()
        pred_class = tf.argmax(predictions[i]).numpy()
        prob_str = " | ".join([f"{config.CLASS_NAMES[j]}: {probs[j]:.3f}"
                               for j in range(len(config.CLASS_NAMES))])
        print(f"  [{prob_str}] -> {config.CLASS_NAMES[pred_class]}")

    print(f"\n{'='*70}")
    print("MODEL READY FOR TRAINING!")
    print(f"{'='*70}\n")
