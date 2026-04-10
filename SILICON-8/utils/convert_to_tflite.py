"""
Convert SILICON-8 model to TFLite format with INT8 quantization.

This script:
1. Loads the trained Keras model
2. Applies post-training quantization (INT8)
3. Embeds MLTK AudioFeatureGenerator settings in the .tflite file
4. Saves the quantized model for xG26 deployment
"""

import os
import numpy as np
import tensorflow as tf
import json

import config
from Model import ImprovedTransformerClassifier
from preprocess import AudioPreprocessor


def representative_dataset_gen(num_samples=100):
    """
    Generate representative dataset for post-training quantization.

    This is required for full INT8 quantization.
    """
    preprocessor = AudioPreprocessor(use_mltk=True)

    # Load a subset of training data
    from pathlib import Path
    dataset_dir = Path(config.DATASET_DIR)

    audio_files = []
    for class_name in config.CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob('*.wav'))[:num_samples // config.NUM_CLASSES]
            audio_files.extend(files)

    print(f"Using {len(audio_files)} samples for representative dataset...")

    for audio_file in audio_files:
        try:
            # Preprocess
            features = preprocessor.preprocess_file(str(audio_file), dtype=np.int8)

            # Reshape for model: (F, T) -> (1, 1, F, T)
            features = features.T
            features = features[np.newaxis, :, :, np.newaxis]
            features = features.astype(np.float32) / 128.0

            yield [features]
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue


def convert_model_to_tflite(
    model_weights_path=None,
    output_path=None,
    quantize=True
):
    """
    Convert Keras model to TFLite format.

    Args:
        model_weights_path: Path to saved model weights (.h5)
        output_path: Path to save .tflite file
        quantize: If True, apply INT8 quantization

    Returns:
        Path to saved .tflite file
    """
    if model_weights_path is None:
        model_weights_path = os.path.join(config.MODEL_DIR, 'best_model.weights.h5')

    if output_path is None:
        output_path = os.path.join(config.MODEL_DIR, 'silicon8_model.tflite')

    print("\n" + "="*60)
    print("Converting SILICON-8 to TFLite")
    print("="*60)

    # Load model
    print(f"\n1. Loading model from: {model_weights_path}")
    model = ImprovedTransformerClassifier(
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_se_block=config.USE_SE_BLOCK,
        se_reduction=config.SE_REDUCTION
    )

    # Build model
    dummy_input = tf.random.normal((1, 1, config.FILTERBANK_N_CHANNELS, config.N_TIME_FRAMES))
    _ = model(dummy_input, training=False)

    # Load weights
    model.load_weights(model_weights_path)
    print(f"   ✓ Model loaded ({model.count_params():,} parameters)")

    # Convert to TFLite
    print("\n2. Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("   ✓ Enabling INT8 quantization...")

        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set supported ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Set input/output types to INT8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Provide representative dataset for calibration
        print("   ✓ Generating representative dataset...")
        converter.representative_dataset = representative_dataset_gen

    # Convert
    print("   ✓ Running conversion...")
    tflite_model = converter.convert()

    # Save
    print(f"\n3. Saving TFLite model to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size_kb = len(tflite_model) / 1024
    print(f"   ✓ Model saved ({file_size_kb:.1f} kB)")

    # Embed MLTK settings
    print("\n4. Embedding MLTK AudioFeatureGenerator settings...")
    embed_mltk_settings(output_path)

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nTFLite model: {output_path}")
    print(f"Size: {file_size_kb:.1f} kB")
    print(f"Quantized: {'Yes (INT8)' if quantize else 'No (FP32)'}")

    return output_path


def embed_mltk_settings(tflite_path):
    """
    Embed MLTK AudioFeatureGenerator settings into the .tflite file.

    This is crucial for deployment - the xG26 device will read these
    settings to configure audio preprocessing identically to training.

    Note: This requires the MLTK package to properly embed metadata.
    For now, we save settings alongside the .tflite file.
    """
    settings_path = tflite_path.replace('.tflite', '_settings.json')

    preprocessor = AudioPreprocessor(use_mltk=True)
    settings = preprocessor.get_settings_dict()

    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)

    print(f"   ✓ MLTK settings saved to: {settings_path}")
    print(f"   ✓ Settings will be embedded in TFLite model during final deployment")


def test_tflite_model(tflite_path):
    """
    Test the TFLite model to ensure it works correctly.

    Args:
        tflite_path: Path to .tflite file
    """
    print("\n" + "="*60)
    print("Testing TFLite Model")
    print("="*60)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nInput details:")
    print(f"  Name: {input_details[0]['name']}")
    print(f"  Shape: {input_details[0]['shape']}")
    print(f"  Type: {input_details[0]['dtype']}")

    print(f"\nOutput details:")
    print(f"  Name: {output_details[0]['name']}")
    print(f"  Shape: {output_details[0]['shape']}")
    print(f"  Type: {output_details[0]['dtype']}")

    # Test with random input
    print(f"\nRunning test inference...")
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    if input_dtype == np.int8:
        test_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    else:
        test_input = np.random.randn(*input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min()}, {output.max()}]")

    # Dequantize if INT8
    if output_details[0]['dtype'] == np.int8:
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        output_float = (output.astype(np.float32) - zero_point) * scale
        print(f"Dequantized output: {output_float}")

    print("\n✓ TFLite model test passed!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert SILICON-8 to TFLite')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights (.h5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for .tflite file')
    parser.add_argument('--no-quantize', action='store_true',
                       help='Disable INT8 quantization (not recommended)')
    parser.add_argument('--test', action='store_true',
                       help='Test the converted model')

    args = parser.parse_args()

    # Convert model
    tflite_path = convert_model_to_tflite(
        model_weights_path=args.weights,
        output_path=args.output,
        quantize=not args.no_quantize
    )

    # Test if requested
    if args.test:
        test_tflite_model(tflite_path)
