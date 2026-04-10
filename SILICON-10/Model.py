import tensorflow as tf
import numpy as np
import math

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Shape (1, max_len, d_model)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x, training=False):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)


class SqueezeExcitation(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation block for channel attention.
    Helps model focus on most discriminative frequency channels.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Squeeze: Global average pooling (done in call)
        # Excitation: Two FC layers
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, x, training=False):
        """
        Args:
            x: (B, H, W, C) format tensor
        Returns:
            Reweighted tensor of same shape
        """
        # Squeeze: Global average pooling across spatial dimensions
        # (B, H, W, C) -> (B, C)
        squeeze = tf.reduce_mean(x, axis=[1, 2])

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)

        # Reshape to (B, 1, 1, C) for broadcasting
        excitation = tf.reshape(excitation, [-1, 1, 1, self.channels])

        # Scale: element-wise multiplication
        return x * excitation


class MultiScaleCNN(tf.keras.layers.Layer):
    """
    Multi-scale CNN with parallel paths capturing different temporal patterns.
    - Branch 1: 3x3 kernels (fine details - short bursts)
    - Branch 2: 5x5 kernels (mid-range patterns)
    - Branch 3: 7x7 kernels (broad patterns - sustained sounds)
    """
    def __init__(self, filters_3x3=4, filters_5x5=2, filters_7x7=2, dropout=0.1):
        super().__init__()
        self.total_filters = filters_3x3 + filters_5x5 + filters_7x7

        # Branch 1: Fine details (3x3)
        self.conv_3x3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters_3x3, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Branch 2: Mid-range (5x5)
        self.conv_5x5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters_5x5, kernel_size=(5, 5), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Branch 3: Broad patterns (7x7)
        self.conv_7x7 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters_7x7, kernel_size=(7, 7), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        """
        Args:
            x: (B, H, W, C) input tensor
        Returns:
            Concatenated multi-scale features
        """
        # Parallel branches
        feat_3x3 = self.conv_3x3(x, training=training)
        feat_5x5 = self.conv_5x5(x, training=training)
        feat_7x7 = self.conv_7x7(x, training=training)

        # Concatenate along channel dimension
        out = tf.concat([feat_3x3, feat_5x5, feat_7x7], axis=-1)

        return self.dropout(out, training=training)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Transformer encoder layer with post-layer normalization.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model // nhead,
            dropout=dropout
        )

        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(d_model)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, src, training=False):
        # Self Attention with residual connection
        src2 = self.self_attn(src, src, training=training)
        src = src + self.dropout1(src2, training=training)
        src = self.norm1(src)

        # Feed Forward with residual connection
        src2 = self.linear2(self.dropout(self.linear1(src), training=training))
        src = src + self.dropout2(src2, training=training)
        src = self.norm2(src)
        return src


class ImprovedTransformerClassifier(tf.keras.Model):
    """
    Production-scale improved classifier with:
    - Multi-scale CNN for diverse temporal patterns
    - Squeeze-Excitation for channel attention
    - Deeper transformer (3 layers) for better temporal modeling
    - Optimized for baby cry vs dog sound classification
    """
    def __init__(self, num_classes=2, d_model=24, num_heads=4, num_layers=3,
                 dropout=0.15, use_se_block=True, se_reduction=4):
        super().__init__()

        self.d_model = d_model
        self.use_se_block = use_se_block

        # Multi-scale CNN: (B, 1, 64, T) -> (B, 64, T, 6)
        # 3 filters @ 3x3 + 2 filters @ 5x5 + 1 filter @ 7x7 = 6 total (reduced to fit budget)
        self.multi_scale_cnn = MultiScaleCNN(
            filters_3x3=3,
            filters_5x5=2,
            filters_7x7=1,
            dropout=dropout
        )

        # Squeeze-and-Excitation block for channel attention
        if use_se_block:
            total_channels = 6  # 3 + 2 + 1 from multi-scale CNN
            self.se_block = SqueezeExcitation(channels=total_channels, reduction=se_reduction)

        # Input projection: (B, T, 64*6=384) -> (B, T, d_model)
        self.input_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # Positional encoding (max_len=1200 to handle variable-length audio up to 10s)
        # Max observed: 1001 frames, setting to 1200 for safety
        self.pos_encoding = PositionalEncoding(d_model, max_len=1200, dropout=dropout)

        # Deeper transformer encoder (3 layers for better capacity)
        self.enc_layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='relu'
            ) for _ in range(num_layers)
        ]

        # Classification head
        self.classifier_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape (B, 1, 64, T)
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Convert to NHWC: (B, 1, 64, T) -> (B, 64, T, 1)
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # Multi-scale CNN: (B, 64, T, 1) -> (B, 64, T, 8)
        x = self.multi_scale_cnn(x, training=training)

        # Squeeze-and-Excitation (channel attention)
        if self.use_se_block:
            x = self.se_block(x, training=training)

        # Reshape: (B, 64, T, 6) -> (B, T, 384)
        B = tf.shape(x)[0]
        T = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, T, 64, 6)
        x = tf.reshape(x, [B, T, -1])  # (B, T, 384)

        # Project to d_model: (B, T, 384) -> (B, T, d_model)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Add positional encoding
        x = self.pos_encoding(x, training=training)

        # Transformer encoding (3 layers)
        for layer in self.enc_layers:
            x = layer(x, training=training)

        # Global mean pooling over time: (B, T, d_model) -> (B, d_model)
        x = tf.reduce_mean(x, axis=1)

        # Classifier: (B, d_model) -> (B, num_classes)
        x = self.classifier_head(x, training=training)

        return x


def count_parameters(model):
    """Count trainable parameters"""
    return model.count_params()


def test_model_initialization():
    """Test that model produces different outputs for different inputs"""
    model = ImprovedTransformerClassifier()

    # Create two different random inputs
    x1 = tf.random.normal((1, 1, 64, 128))
    x2 = tf.random.normal((1, 1, 64, 128))

    # Call once to build
    out1 = model(x1, training=False)
    out2 = model(x2, training=False)

    # Check if outputs are different
    diff = tf.reduce_max(tf.abs(out1 - out2))
    if diff < 1e-5:
        print("❌ ERROR: Model outputs are identical!")
        return False
    else:
        print("✓ Model produces different outputs for different inputs")
        print(f" Output 1: {out1.numpy()}")
        print(f" Output 2: {out2.numpy()}")
        print(f" Difference: {diff.numpy():.6f}")
        return True


if __name__ == '__main__':
    # Production-scale configuration (optimized to fit under 17K)
    TEST_D_MODEL = 18
    TEST_NUM_HEADS = 3
    TEST_NUM_LAYERS = 3
    TEST_DROPOUT = 0.15
    TEST_NUM_CLASSES = 4  # Updated for 4-class classification
    TEST_MAX_PARAMS = 17000
    TEST_USE_SE = True  # Enabled - fits within budget!

    print("="*60)
    print("IMPROVED PRODUCTION-SCALE MODEL")
    print("="*60)
    print(f"Configuration:")
    print(f"  d_model: {TEST_D_MODEL}")
    print(f"  num_heads: {TEST_NUM_HEADS}")
    print(f"  num_layers: {TEST_NUM_LAYERS}")
    print(f"  dropout: {TEST_DROPOUT}")
    print(f"  SE Block: {TEST_USE_SE}")
    print(f"  Multi-scale CNN: 3x3(3) + 5x5(2) + 7x7(1)")
    print("="*60)

    model = ImprovedTransformerClassifier(
        num_classes=TEST_NUM_CLASSES,
        d_model=TEST_D_MODEL,
        num_heads=TEST_NUM_HEADS,
        num_layers=TEST_NUM_LAYERS,
        dropout=TEST_DROPOUT,
        use_se_block=TEST_USE_SE
    )

    # Build the model with a dummy input
    dummy = tf.random.normal((1, 1, 64, 128))
    _ = model(dummy)

    params = count_parameters(model)
    print(f'\nTotal parameters: {params:,}')
    print(f'Parameter budget: {TEST_MAX_PARAMS:,}')

    if params <= TEST_MAX_PARAMS:
        print(f'✓ Within budget! ({TEST_MAX_PARAMS - params:,} parameters remaining)')
    else:
        print(f'❌ Exceeds budget by {params - TEST_MAX_PARAMS:,} parameters')

    # Test forward pass with fixed length
    print('\n--- Testing Fixed Length Input ---')
    x_fixed = tf.random.normal((2, 1, 64, 128))
    out = model(x_fixed)
    print(f'Input shape: {x_fixed.shape}')
    print(f'Output shape: {out.shape}')

    # Test forward pass with variable length
    print('\n--- Testing Variable Length Input ---')
    x_var = tf.random.normal((2, 1, 64, 200))
    out_var = model(x_var)
    print(f'Input shape: {x_var.shape}')
    print(f'Output shape: {out_var.shape}')

    # Test initialization
    print('\n--- Testing Model Initialization ---')
    test_model_initialization()

    # Check that different batches produce different outputs
    print("\n--- Testing Batch Outputs ---")
    out = model(x_fixed, training=False)
    print(f"Batch output 0: {out[0].numpy()}")
    print(f"Batch output 1: {out[1].numpy()}")

    is_diff = not np.allclose(out[0], out[1], atol=1e-5)
    print(f"Outputs are different: {is_diff}")

    print("\n" + "="*60)
    print("MODEL VALIDATION COMPLETE")
    print("="*60)
