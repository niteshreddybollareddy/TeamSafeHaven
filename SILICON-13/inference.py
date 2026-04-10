"""
SILICON-13 Inference Script
3-Class Classification: Baby Cry, Dog, Unknown
Based on SILICON-9 inference pipeline
"""

import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
import scipy.signal
from pathlib import Path

import config
from model import CryNet3Class


class AudioInference:
    """
    Real-time inference engine for SILICON-13 audio classifier.
    Based on SILICON-9 inference with 3-class support.
    """

    def __init__(self, model_path, stats_path=None):
        """
        Initialize inference engine.

        Args:
            model_path: Path to saved model weights
            stats_path: Path to normalization stats
        """
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)

        # Load normalization stats
        if stats_path is None:
            stats_path = os.path.join(self.model_dir, 'stats.npy')

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats['mean']
        self.std = stats['std']

        print(f"Loaded normalization stats:")
        print(f"  Mean: {self.mean:.4f}")
        print(f"  Std:  {self.std:.4f}")

        # Initialize model
        self.model = self._load_model()

        # Class names (3 classes)
        self.class_names = config.CLASS_NAMES

        print(f"\nInference engine ready!")
        print(f"Classes: {self.class_names}")

    def _load_model(self):
        """Load trained model."""
        print(f"\nLoading model from: {self.model_path}")

        model = CryNet3Class(
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )

        # Build model
        dummy_input = tf.zeros((1, 1, 64, 148))
        _ = model(dummy_input, training=False)

        # Load weights
        try:
            model.load_weights(self.model_path)
            print(f"Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        return model

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """Apply DC notch filter."""
        b = [1, -1]
        a = [1, -coefficient]
        return scipy.signal.lfilter(b, a, audio)

    def _apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def _select_max_energy_window(self, audio, target_length):
        """Select window with maximum energy."""
        if len(audio) < target_length:
            pad_length = target_length - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')

        frame_energy = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=512
        ).squeeze()

        max_energy_frame = np.argmax(frame_energy)
        center_sample = max_energy_frame * 512

        start_idx = center_sample - target_length // 2
        start_idx = max(0, min(start_idx, len(audio) - target_length))
        end_idx = start_idx + target_length

        return audio[start_idx:end_idx]

    def load_and_preprocess(self, audio_path):
        """Load and preprocess audio file."""
        # Load audio
        audio, orig_sr = sf.read(audio_path, always_2d=False)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if orig_sr != config.SAMPLE_RATE_HZ:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=config.SAMPLE_RATE_HZ)

        # DC filter
        audio = self._apply_dc_notch_filter(audio)

        # Energy-based windowing
        target_length = config.SAMPLE_LENGTH
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = self._select_max_energy_window(audio, target_length)

        # Light silence trimming
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=40)
        if len(audio_trimmed) < target_length:
            audio = np.pad(audio_trimmed, (0, target_length - len(audio_trimmed)), mode='constant')
        else:
            audio = audio_trimmed[:target_length]

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

    def extract_features(self, audio):
        """Extract mel-spectrogram features."""
        # Pre-emphasis
        audio = self._apply_preemphasis(audio)

        # STFT
        stft = librosa.stft(
            audio,
            n_fft=config.FFT_LENGTH,
            hop_length=config.WINDOW_STEP_LENGTH,
            win_length=config.WINDOW_SIZE_LENGTH,
            window='hann',
            center=True,
            pad_mode='constant'
        )

        # Power spectrogram
        power_spec = np.abs(stft) ** 2

        # Mel filterbank
        mel_basis = librosa.filters.mel(
            sr=config.SAMPLE_RATE_HZ,
            n_fft=config.FFT_LENGTH,
            n_mels=config.FILTERBANK_N_CHANNELS,
            fmin=config.FILTERBANK_LOWER_BAND_LIMIT,
            fmax=config.FILTERBANK_UPPER_BAND_LIMIT,
            htk=False
        )

        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)

        # Noise reduction
        noise_floor = np.median(mel_spec[:, :5], axis=1, keepdims=True)
        mel_spec = mel_spec - noise_floor
        mel_spec = np.maximum(mel_spec, 0)

        # Log scale
        mel_spec_db = librosa.power_to_db(mel_spec + 1e-10, ref=np.max, top_db=80.0)

        # Normalize
        mel_spec_norm = (mel_spec_db - self.mean) / (self.std + 1e-8)

        # Add batch and channel dimensions
        mel_spec_norm = mel_spec_norm[np.newaxis, np.newaxis, ...]
        mel_spec_norm = mel_spec_norm.astype(np.float32)

        return mel_spec_norm

    def predict(self, audio_path):
        """Run inference on audio file."""
        # Preprocess
        audio = self.load_and_preprocess(audio_path)

        # Extract features
        features = self.extract_features(audio)

        # Inference
        probs = self.model(tf.constant(features), training=False).numpy()[0]

        # Get prediction
        class_idx = int(np.argmax(probs))
        class_name = self.class_names[class_idx]
        confidence = float(probs[class_idx])

        return {
            'class_idx': class_idx,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
        }

    def predict_batch(self, audio_paths):
        """Run inference on multiple audio files."""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append(None)

        return results


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='SILICON-13 Audio Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, default='models/best_model.weights.h5',
                        help='Path to model weights')
    parser.add_argument('--stats', type=str, default=None,
                        help='Path to stats file')

    args = parser.parse_args()

    print("="*60)
    print("SILICON-13 Audio Classification Inference")
    print("3-Class: Baby Cry, Dog, Unknown")
    print("="*60)

    inference = AudioInference(args.model, args.stats)

    audio_path = Path(args.audio)

    if audio_path.is_file():
        print(f"\nProcessing: {audio_path}")
        result = inference.predict(str(audio_path))

        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Predicted Class: {result['class_name']}")
        print(f"Confidence:      {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"\nAll Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name:12s}: {prob:.4f} ({prob*100:.2f}%)")
        print("="*60)

    elif audio_path.is_dir():
        audio_files = list(audio_path.glob('*.wav')) + list(audio_path.glob('*.ogg'))

        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return

        print(f"\nProcessing {len(audio_files)} files from: {audio_path}")

        results = inference.predict_batch([str(f) for f in audio_files])

        print("\n" + "="*60)
        print("BATCH PREDICTION RESULTS")
        print("="*60)

        for audio_file, result in zip(audio_files, results):
            if result is not None:
                print(f"{audio_file.name:40s} -> {result['class_name']:12s} ({result['confidence']*100:5.2f}%)")
            else:
                print(f"{audio_file.name:40s} -> ERROR")

        print("\n" + "="*60)
        print("PREDICTION DISTRIBUTION")
        print("="*60)

        valid_results = [r for r in results if r is not None]
        for class_name in inference.class_names:
            count = sum(1 for r in valid_results if r['class_name'] == class_name)
            pct = count / len(valid_results) * 100 if valid_results else 0
            print(f"{class_name:12s}: {count:4d} files ({pct:5.1f}%)")

        print("="*60)

    else:
        print(f"Error: {audio_path} is not a valid file or directory")


if __name__ == '__main__':
    main()
