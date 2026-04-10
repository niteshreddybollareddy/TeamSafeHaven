"""
SILICON-22: CryNet Architecture (MVP Compatible)
- Problem-driven acoustic architecture optimized for Silicon Labs xG26.
- Three semantic pillars: Envelope, Spectral, and Temporal Pattern.
- Optimized for INT8 quantization and hardware acceleration.
- No dilations, no custom ops, no CPU fallback.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(64, 198, 1), num_classes=3, training=False):
    """
    CryNet V2: MVP-Optimized Triple-Flow Architecture
    - All layers are compatible with the xG26 hardware accelerator.
    - No dilation_rate usage.
    - All activations are ReLU.
    - Channel counts are multiples of 8 for alignment.
    """
    
    inputs = layers.Input(shape=input_shape, name="data")

    # ---------------------------------------------------------
    # PILLAR 1: ENVELOPE PILLAR (Energy Evolution)
    # Focus: Temporal dynamics using (1, K) kernels and time-pooling.
    # ---------------------------------------------------------
    x_env = layers.Conv2D(16, (1, 5), padding='same', activation='relu')(inputs)
    x_env = layers.MaxPooling2D(pool_size=(1, 2))(x_env)   # Downsample time

    x_env = layers.Conv2D(24, (1, 5), padding='same', activation='relu')(x_env)
    x_env = layers.MaxPooling2D(pool_size=(1, 2))(x_env)

    x_env = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(x_env)
    x_env_feat = layers.GlobalAveragePooling2D()(x_env)

    # ---------------------------------------------------------
    # PILLAR 2: SPECTRAL PILLAR (Frequency Structure)
    # Focus: Harmonic spacing using (K, 1) kernels and frequency-pooling.
    # ---------------------------------------------------------
    x_spec = layers.Conv2D(16, (5, 1), padding='same', activation='relu')(inputs)
    x_spec = layers.MaxPooling2D(pool_size=(2, 1))(x_spec)

    x_spec = layers.Conv2D(24, (3, 1), padding='same', activation='relu')(x_spec)
    x_spec = layers.MaxPooling2D(pool_size=(2, 1))(x_spec)

    x_spec = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(x_spec)
    x_spec_feat = layers.GlobalAveragePooling2D()(x_spec)

    # ---------------------------------------------------------
    # PILLAR 3: TEMPORAL PATTERN PILLAR (Rhythm/Repetition)
    # Focus: Long-term periodicity using larger kernels.
    # ---------------------------------------------------------
    x_temp = layers.Conv2D(16, (1, 7), padding='same', activation='relu')(inputs)
    x_temp = layers.MaxPooling2D(pool_size=(1, 2))(x_temp)

    x_temp = layers.Conv2D(24, (1, 9), padding='same', activation='relu')(x_temp)
    x_temp = layers.MaxPooling2D(pool_size=(1, 2))(x_temp)

    x_temp = layers.Conv2D(32, (1, 11), padding='same', activation='relu')(x_temp)
    x_temp_feat = layers.GlobalAveragePooling2D()(x_temp)

    # ---------------------------------------------------------
    # FUSION ENGINE
    # Combines features from all three pillars into a unified decision.
    # ---------------------------------------------------------
    # Concatenate vectors (GlobalAveragePooling2D already used on each pillar)
    merged = layers.Concatenate()([x_env_feat, x_spec_feat, x_temp_feat])
    
    # Final Classification Head
    # Mix concatenated features (using Dense instead of Conv2D 1x1 on maps to handle shape mismatch)
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dense(32, activation='relu')(x)
    
    output = layers.Dense(num_classes, activation='softmax', name='main_output')(x)

    # Final Model Construction
    # Silicon-22 focuses on single-output inference and standard training to ensure MVP compatibility.
    # Auxiliary heads from CryNet V1 are removed.
    model = models.Model(inputs=inputs, outputs=output)
        
    return model

if __name__ == '__main__':
    # Test inference build
    model = create_model()
    model.summary()
    print("\nModel build successful. Check summary for MVP-compatible layer shapes.")
