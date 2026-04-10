import tensorflow as tf
import numpy as np
from model import CryNetMicroV2

# Load model
model = CryNetMicroV2(num_classes=2, dropout=0.3)
model.build((None, 1, 64, 98))
model.load_weights('experiments/crynet_micro_v2_20260111_182431/checkpoints/best_model.weights.h5')

# Create concrete function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1, 64, 98], dtype=tf.float32)])
def model_fn(x):
    return model(x, training=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([model_fn.get_concrete_function()])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('experiments/crynet_micro_v2_20260111_182431/checkpoints/model.tflite', 'wb') as f:
    f.write(tflite_model)

print('✓ TFLite model created')
print(f'  Size: {len(tflite_model)} bytes')
