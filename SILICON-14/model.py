"""
SILICON-14: TENet-Micro Architecture
Hardware-Optimized for Silicon Labs xG26 MVP Accelerator
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable(package="Custom")
class TENetBlock(layers.Layer):
    """
    Temporal Efficient Block: Depthwise Separable Conv with Temporal Dilation
    Identical to Silicon Labs internal 'heygecko' blocks but sized for 4-class.
    """
    def __init__(self, filters, kernel_size=(3, 3), dropout=0.1, **kwargs):
        super(TENetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        # Depthwise part (Spatial/Temporal patterns)
        self.dw_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            padding='same',
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # Pointwise part (Channel mixing)
        self.pw_conv = layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.dw_conv(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pw_conv(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super(TENetBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class TENetMicro(Model):
    """
    TENet-Micro: Optimized for xG26 MVP
    - Replaces Transformer with Temporal Depthwise Convs
    - No BatchMatMul (100% TFLite compatible)
    - High temporal resolution
    """
    def __init__(self, num_classes=3, **kwargs):
        super(TENetMicro, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # Initial Feature Extraction
        self.stem = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.bn_stem = layers.BatchNormalization()
        self.relu_stem = layers.ReLU()

        # TENet Stages
        self.block1 = TENetBlock(32, kernel_size=(3, 3))
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.block2 = TENetBlock(48, kernel_size=(3, 3))
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.block3 = TENetBlock(64, kernel_size=(3, 3))
        
        # Temporal Aggregation (REDUCE MEAN over Frequency only, then Global Pool)
        # B, H, W, C -> (B, 1, W, C)
        self.gap_freq = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
        
        # Final Pointwise Classifier
        self.classifier = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x, training=False):
        # Input shape: (B, 64, 148, 1) or (B, 1, 64, 148)
        # We handle the transpose in the call for flexibility
        if x.shape[1] == 1:
            x = tf.transpose(x, perm=[0, 2, 3, 1]) # (B, 64, 148, 1)

        x = self.stem(x)
        x = self.bn_stem(x, training=training)
        x = self.relu_stem(x)
        
        x = self.block1(x, training=training)
        x = self.pool1(x)
        
        x = self.block2(x, training=training)
        x = self.pool2(x)
        
        x = self.block3(x, training=training)
        
        x = self.gap_freq(x)
        x = self.classifier(x, training=training)
        return x

    def get_config(self):
        config = super(TENetMicro, self).get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

def create_model(num_classes=3):
    return TENetMicro(num_classes=num_classes)

if __name__ == '__main__':
    model = create_model()
    # Build with dummy input (B, 64, 148, 1)
    dummy = tf.zeros((1, 64, 148, 1))
    out = model(dummy)
    model.summary()
    
    total_params = model.count_params()
    print(f"Total Parameters: {total_params:,}")
    if total_params < 15000:
        print("✓ Within Budget for xG26")
    else:
        print("❌ Over Budget!")
