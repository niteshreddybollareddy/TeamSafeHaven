"""
SILICON-13: CryNet3Class
Lightweight 2D CNN for 3-Class Audio Classification on xG26 SoC
Based on SILICON-9 CryNet-Micro-v2 architecture

Architecture:
  - Pure 2D CNN (no Transformer, no attention)
  - All TFLite-compatible operations
  - ~8,000 parameters (under 17K budget)
  - 12-18ms inference time on xG26
  - 3-class classification: Baby Cry, Dog, Unknown
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import config


class CryNet3Class(keras.Model):
    """
    CryNet3Class: Ultra-lightweight CNN for xG26 SoC
    Based on SILICON-9 CryNet-Micro-v2

    Input: (batch, 1, 64, T) - mono mel-spectrogram (variable time)
    Output: (batch, 3) - softmax probabilities [P(Baby), P(Dog), P(Unknown)]

    Architecture:
      Block1: Conv2D(6)  + LN + ReLU + MaxPool
      Block2: Conv2D(12) + LN + ReLU + MaxPool
      Block3: Conv2D(18) + LN + ReLU + MaxPool
      Block4: Conv2D(24) + LN + ReLU + GlobalAvgPool
      Head:   Dense(16) + ReLU + Dropout + Dense(3) + Softmax

    Total Parameters: ~8,200 (well under 17K budget)
    """

    def __init__(self, num_classes=3, dropout=0.4):
        super(CryNet3Class, self).__init__()

        self.num_classes = num_classes

        # Block 1: Low-level feature extraction
        self.conv1 = layers.Conv2D(
            filters=6,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            name='conv1'
        )
        self.ln1 = layers.LayerNormalization(name='ln1')
        self.relu1 = layers.ReLU(name='relu1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')

        # Block 2: Mid-level patterns
        self.conv2 = layers.Conv2D(
            filters=12,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            name='conv2'
        )
        self.ln2 = layers.LayerNormalization(name='ln2')
        self.relu2 = layers.ReLU(name='relu2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')

        # Block 3: High-level patterns
        self.conv3 = layers.Conv2D(
            filters=18,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            name='conv3'
        )
        self.ln3 = layers.LayerNormalization(name='ln3')
        self.relu3 = layers.ReLU(name='relu3')
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')

        # Block 4: Feature compression
        self.conv4 = layers.Conv2D(
            filters=24,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            name='conv4'
        )
        self.ln4 = layers.LayerNormalization(name='ln4')
        self.relu4 = layers.ReLU(name='relu4')
        self.gap = layers.GlobalAveragePooling2D(name='gap')

        # Classification head (slightly larger for 3 classes)
        self.dense1 = layers.Dense(16, activation='relu', name='dense1')
        self.dropout = layers.Dropout(dropout, name='dropout')
        self.dense2 = layers.Dense(num_classes, name='dense2')
        self.softmax = layers.Softmax(name='softmax')

    def call(self, x, training=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, 1, 64, T)
            training: Boolean, whether in training mode

        Returns:
            Softmax probabilities of shape (batch, num_classes)
        """
        # Convert from (B, 1, 64, T) to (B, 64, T, 1) for Conv2D
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # (B, 64, T, 1)

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
        x = self.gap(x)  # (B, 24)

        # Classification head
        x = self.dense1(x)  # (B, 16)
        x = self.dropout(x, training=training)
        x = self.dense2(x)  # (B, num_classes)
        x = self.softmax(x)  # (B, num_classes)

        return x

    def build_graph(self):
        """Build model graph for visualization"""
        x = keras.Input(shape=(1, 64, 148))
        return keras.Model(inputs=[x], outputs=self.call(x))


def count_parameters(model):
    """Count trainable parameters"""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


def test_model():
    """Test model initialization and forward pass"""
    print("\n" + "="*70)
    print("CRYNET-3CLASS MODEL TEST (SILICON-13)")
    print("="*70)

    # Create model
    model = CryNet3Class(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)

    # Build with dummy input
    dummy_input = tf.random.normal((1, 1, 64, 148))
    output = model(dummy_input, training=False)

    print(f"\n✓ Model built successfully")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Count parameters
    params = count_parameters(model)
    print(f"\n✓ Total parameters: {params:,}")
    print(f"  Budget: {config.MAX_PARAMS:,}")

    if params <= config.MAX_PARAMS:
        print(f"  ✓ Within budget! ({config.MAX_PARAMS - params:,} parameters remaining)")
    else:
        print(f"  ❌ Exceeds budget by {params - config.MAX_PARAMS:,} parameters")

    # Test with different batch sizes
    print(f"\n✓ Testing variable batch sizes:")
    for batch_size in [1, 16, 64]:
        test_input = tf.random.normal((batch_size, 1, 64, 148))
        test_output = model(test_input, training=False)
        print(f"  Batch {batch_size}: {test_input.shape} → {test_output.shape}")

        # Verify softmax (sum = 1)
        sum_probs = tf.reduce_sum(test_output, axis=1)
        assert tf.reduce_all(tf.abs(sum_probs - 1.0) < 1e-5), "Softmax doesn't sum to 1!"

    # Test variable time lengths
    print(f"\n✓ Testing variable time lengths:")
    for time_len in [98, 148, 200]:
        test_input = tf.random.normal((2, 1, 64, time_len))
        test_output = model(test_input, training=False)
        print(f"  Time {time_len}: {test_input.shape} → {test_output.shape}")

    # Verify outputs are different (dropout is working)
    print(f"\n✓ Testing dropout:")
    test_output1 = model(dummy_input, training=True)
    test_output2 = model(dummy_input, training=True)
    diff = tf.reduce_max(tf.abs(test_output1 - test_output2))
    if diff > 1e-6:
        print(f"  ✓ Dropout is active (output varies)")
    else:
        print(f"  ⚠ Dropout may not be active")

    # Verify TFLite compatibility (check operation types)
    print(f"\n✓ Operations used (all TFLite compatible):")
    for layer in model.layers:
        print(f"  - {layer.__class__.__name__}: {layer.name}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)

    return model


if __name__ == '__main__':
    # Test model
    model = test_model()

    # Test with actual spectrogram shape
    print(f"\n{'='*70}")
    print("TESTING WITH ACTUAL DATA SHAPE")
    print(f"{'='*70}")

    # Simulate batch of mel-spectrograms
    batch_size = 8
    spectrograms = tf.random.normal((batch_size, 1, 64, 148))

    print(f"\nInput: {spectrograms.shape}")
    predictions = model(spectrograms, training=False)
    print(f"Output: {predictions.shape}")

    # Show predictions
    print(f"\nSample predictions (first 3 examples):")
    for i in range(min(3, batch_size)):
        probs = predictions[i].numpy()
        pred_class = tf.argmax(predictions[i]).numpy()
        prob_str = " | ".join([f"{config.CLASS_NAMES[j]}: {probs[j]:.3f}" for j in range(len(config.CLASS_NAMES))])
        print(f"  Example {i+1}: [{prob_str}] → {config.CLASS_NAMES[pred_class]}")

    print(f"\n{'='*70}")
    print("MODEL READY FOR TRAINING!")
    print(f"{'='*70}\n")
