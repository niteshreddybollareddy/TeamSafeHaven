"""
SILICON-10: Lightweight 2D CNN for Baby Cry Detection
Chip-friendly architecture with:
- Pure 2D CNN (no transformers, no unsupported ops)
- Fast inference (<20ms target)
- Supports binary (1 vs rest) and multi-class training
- MVP design - simple but effective
"""

import tensorflow as tf
import numpy as np


class BabyCryDetectorCNN(tf.keras.Model):
    """
    Lightweight 2D CNN optimized for embedded deployment

    Architecture Philosophy:
    - Small kernels (3x3) for efficiency
    - Depthwise separable convs for reduced parameters
    - LayerNorm instead of BatchNorm (more stable, chip-friendly)
    - Progressive downsampling (preserve early features)
    - Global average pooling (reduce overfitting)

    Target: <10K parameters, <20ms inference
    """

    def __init__(self, num_classes=2, dropout=0.2, use_depthwise=True):
        super().__init__()

        self.num_classes = num_classes
        self.use_depthwise = use_depthwise

        # Block 1: Initial feature extraction (64x148 -> 32x74)
        # Standard conv for first layer (depthwise needs multiple channels)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            name='conv1'
        )
        self.ln1 = tf.keras.layers.LayerNormalization(name='ln1')
        self.act1 = tf.keras.layers.ReLU(name='act1')
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool1'
        )

        # Block 2: Mid-level features (32x74 -> 16x37)
        if use_depthwise:
            # Depthwise separable conv (fewer parameters)
            self.conv2_dw = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv2_dw'
            )
            self.conv2_pw = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv2_pw'
            )
        else:
            self.conv2 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv2'
            )

        self.ln2 = tf.keras.layers.LayerNormalization(name='ln2')
        self.act2 = tf.keras.layers.ReLU(name='act2')
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool2'
        )

        # Block 3: High-level features (16x37 -> 8x18)
        if use_depthwise:
            self.conv3_dw = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv3_dw'
            )
            self.conv3_pw = tf.keras.layers.Conv2D(
                filters=24,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv3_pw'
            )
        else:
            self.conv3 = tf.keras.layers.Conv2D(
                filters=24,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv3'
            )

        self.ln3 = tf.keras.layers.LayerNormalization(name='ln3')
        self.act3 = tf.keras.layers.ReLU(name='act3')
        self.pool3 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool3'
        )

        # Block 4: Final conv (8x18 -> 8x18, no pooling)
        if use_depthwise:
            self.conv4_dw = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv4_dw'
            )
            self.conv4_pw = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv4_pw'
            )
        else:
            self.conv4 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                name='conv4'
            )

        self.ln4 = tf.keras.layers.LayerNormalization(name='ln4')
        self.act4 = tf.keras.layers.ReLU(name='act4')

        # Global Average Pooling (chip-friendly, no flatten issues)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')

        # Classification head
        self.dropout = tf.keras.layers.Dropout(dropout, name='dropout')
        self.fc = tf.keras.layers.Dense(
            num_classes,
            activation=None,  # No activation - FocalLoss handles softmax
            name='output'
        )

    def call(self, x, training=False):
        """
        Forward pass

        Args:
            x: Input tensor (B, 1, 64, T) or (B, 64, T, 1)

        Returns:
            Output probabilities (B, num_classes)
        """
        # Ensure NHWC format: (B, H, W, C)
        if x.shape[1] == 1:  # (B, 1, H, W) -> (B, H, W, 1)
            x = tf.transpose(x, perm=[0, 2, 3, 1])

        # Block 1
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.pool1(x)

        # Block 2
        if self.use_depthwise:
            x = self.conv2_dw(x)
            x = self.conv2_pw(x)
        else:
            x = self.conv2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # Block 3
        if self.use_depthwise:
            x = self.conv3_dw(x)
            x = self.conv3_pw(x)
        else:
            x = self.conv3(x)
        x = self.ln3(x)
        x = self.act3(x)
        x = self.pool3(x)

        # Block 4
        if self.use_depthwise:
            x = self.conv4_dw(x)
            x = self.conv4_pw(x)
        else:
            x = self.conv4(x)
        x = self.ln4(x)
        x = self.act4(x)

        # Global pooling and classification
        x = self.gap(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


if __name__ == '__main__':
    print("="*70)
    print("SILICON-10: LIGHTWEIGHT 2D CNN FOR BABY CRY DETECTION")
    print("="*70)

    # Test configurations
    configs = [
        {"name": "Binary (Baby vs Rest)", "num_classes": 2, "use_depthwise": True},
        {"name": "Multi-class (3 classes)", "num_classes": 3, "use_depthwise": True},
        {"name": "Standard Conv (no depthwise)", "num_classes": 2, "use_depthwise": False},
    ]

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        print(f"Num classes: {config['num_classes']}")
        print(f"Depthwise separable: {config['use_depthwise']}")

        model = BabyCryDetectorCNN(
            num_classes=config['num_classes'],
            dropout=0.2,
            use_depthwise=config['use_depthwise']
        )

        # Build model (2-second audio: 64 mel bins × ~195 time frames)
        dummy_input = tf.random.normal((1, 1, 64, 195))
        output = model(dummy_input, training=False)

        params = count_parameters(model)
        print(f"\nTotal parameters: {params:,}")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output (logits): {output.numpy()}")  # Raw logits, not probabilities

        # Estimate inference time (rough, depends on hardware)
        # Rule of thumb: ~0.1ms per 1K parameters on modern CPU
        estimated_inference_ms = params / 1000 * 0.1
        print(f"Estimated inference: ~{estimated_inference_ms:.1f}ms (CPU)")

        # Check memory footprint
        memory_kb = params * 4 / 1024  # 4 bytes per float32
        print(f"Model memory: ~{memory_kb:.1f} KB")

        if params < 10000:
            print(f"✓ Excellent! Well under 10K parameter target")
        elif params < 15000:
            print(f"✓ Good! Under 15K parameter target")
        else:
            print(f"⚠ Warning: Exceeds recommended parameter budget")

    print("\n" + "="*70)
    print("ARCHITECTURE SUMMARY")
    print("="*70)
    print("""
    Layer Structure (with depthwise separable):

    Input:        (B, 1, 64, 195)  - Mel spectrogram (2-second audio, FIXED LENGTH)

    Block 1:      Conv2D(8, 3x3) + LN + ReLU + MaxPool(2x2)
                  -> (B, 32, 97, 8)

    Block 2:      DepthwiseConv(3x3) + Conv2D(16, 1x1) + LN + ReLU + MaxPool(2x2)
                  -> (B, 16, 48, 16)

    Block 3:      DepthwiseConv(3x3) + Conv2D(24, 1x1) + LN + ReLU + MaxPool(2x2)
                  -> (B, 8, 24, 24)

    Block 4:      DepthwiseConv(3x3) + Conv2D(32, 1x1) + LN + ReLU
                  -> (B, 8, 24, 32)

    Head:         GlobalAvgPool + Dropout + Dense(num_classes)
                  -> (B, num_classes) [logits, not probabilities]

    Key Features:
    - Pure 2D CNN (no transformers, no attention)
    - Depthwise separable convs (reduce parameters by ~9x)
    - LayerNorm (more stable than BatchNorm)
    - Progressive downsampling (2x2 pooling)
    - Global average pooling (prevent overfitting)
    - All operations supported on embedded chips

    Target Performance:
    - Parameters: <10K
    - Inference: <20ms
    - Memory: <50KB
    """)

    print("="*70)
    print("READY FOR TRAINING")
    print("="*70)
