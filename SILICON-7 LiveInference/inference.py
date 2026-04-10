"""
SILICON-8 Inference Script with MLTK Preprocessing
Updated to use MLTK AudioFeatureGenerator approach for xG26 deployment

This script provides real-time inference for the 4-class audio classifier:
  - Class 0: Baby Cry
  - Class 1: Dog
  - Class 2: Unknown
  - Class 3: Unusual

Now uses MLTK preprocessing for 100% train/deploy match!
"""

import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
import scipy.signal
import librosa
from pathlib import Path

# Import model and config
import config
from Model import ImprovedTransformerClassifier


class AudioInference:
    """
    Real-time inference engine with MLTK preprocessing.

    Features:
      - MLTK AudioFeatureGenerator preprocessing
      - Hardware-compatible DSP (same as training)
      - Variable-length audio support
      - Automatic normalization using training stats
    """

    def __init__(self, model_path, stats_path=None, max_duration_sec=10.0):
        """
        Initialize inference engine.

        Args:
            model_path (str): Path to saved model weights (.weights.h5)
            stats_path (str): Path to normalization stats (.npy).
                            If None, looks for 'stats.npy' in model directory.
            max_duration_sec (float): Maximum audio duration in seconds (default: 10.0)
        """
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)

        # MLTK preprocessing parameters
        self.sample_rate = config.SAMPLE_RATE_HZ
        self.window_size_ms = config.WINDOW_SIZE_MS
        self.window_step_ms = config.WINDOW_STEP_MS
        self.n_fft = config.FFT_LENGTH
        self.n_mels = config.FILTERBANK_N_CHANNELS
        self.fmin = config.FILTERBANK_LOWER_BAND_LIMIT
        self.fmax = config.FILTERBANK_UPPER_BAND_LIMIT
        self.noise_reduction = config.NOISE_REDUCTION_ENABLE
        self.dc_filter = config.DC_NOTCH_FILTER_ENABLE

        # Maximum duration (trim longer audio to avoid memory issues)
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(self.max_duration_sec * self.sample_rate)

        # Calculate frame parameters
        self.window_size_length = int((self.window_size_ms * self.sample_rate) / 1000)
        self.window_step_length = int((self.window_step_ms * self.sample_rate) / 1000)

        # Load normalization stats
        if stats_path is None:
            stats_path = os.path.join(self.model_dir, 'stats.npy')

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats['mean']
        self.std = stats['std']

        print(f"✓ MLTK Preprocessing initialized")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Max duration: {self.max_duration_sec}s ({self.max_samples} samples)")
        print(f"  Window size: {self.window_size_ms} ms ({self.window_size_length} samples)")
        print(f"  Window step: {self.window_step_ms} ms ({self.window_step_length} samples)")
        print(f"  FFT length: {self.n_fft}")
        print(f"  Mel bands: {self.n_mels}")
        print(f"  Frequency range: {self.fmin}-{self.fmax} Hz")
        print(f"\n✓ Normalization stats loaded:")
        print(f"  Mean: {self.mean:.4f}")
        print(f"  Std:  {self.std:.4f}")

        # Initialize model
        self.model = self._load_model()

        # Class names
        self.class_names = config.CLASS_NAMES

        print(f"\n✓ Inference engine ready!")
        print(f"  Classes: {self.class_names}")

    def _load_model(self):
        """
        Load trained model with saved weights.

        Returns:
            ImprovedTransformerClassifier: Loaded model
        """
        print(f"\nLoading model from: {self.model_path}")

        # Initialize model with same config as training
        model = ImprovedTransformerClassifier(
            num_classes=config.NUM_CLASSES,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            use_se_block=config.USE_SE_BLOCK,
            se_reduction=config.SE_REDUCTION
        )

        # Build model with dummy input
        dummy_input = tf.zeros((1, 1, 64, 128))
        _ = model(dummy_input, training=False)

        # Load weights
        try:
            model.load_weights(self.model_path)
            print(f"✓ Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        return model

    # =========================================================================
    # MLTK PREPROCESSING FUNCTIONS
    # =========================================================================

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """
        Apply DC notch filter to remove DC offset (MLTK feature).
        H(z) = (1 - z^-1) / (1 - coefficient*z^-1)

        Args:
            audio: Input audio signal
            coefficient: Filter coefficient (default: 0.95)

        Returns:
            Filtered audio
        """
        b = [1, -1]  # Numerator coefficients
        a = [1, -coefficient]  # Denominator coefficients
        filtered = scipy.signal.lfilter(b, a, audio)
        return filtered

    def _apply_preemphasis(self, audio, coeff=0.97):
        """
        Apply pre-emphasis filter (boosts high frequencies).

        Args:
            audio: Input audio signal
            coeff: Pre-emphasis coefficient

        Returns:
            Pre-emphasized signal
        """
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def _apply_noise_reduction(self, mel_spec):
        """
        Simple noise reduction by subtracting noise floor estimate (MLTK feature).

        Args:
            mel_spec: Mel spectrogram (mel_bins, time_frames)

        Returns:
            Noise-reduced mel spectrogram
        """
        # Estimate noise floor from first few frames
        noise_floor = np.median(mel_spec[:, :5], axis=1, keepdims=True)

        # Subtract noise floor
        mel_spec_clean = mel_spec - noise_floor

        # Ensure minimum signal remaining
        mel_spec_clean = np.maximum(mel_spec_clean, mel_spec * config.NOISE_REDUCTION_MIN_SIGNAL_REMAINING)

        return mel_spec_clean

    def load_and_preprocess_audio(self, audio_path):
        """
        Load audio file using MLTK approach.

        Args:
            audio_path (str): Path to audio file

        Returns:
            np.ndarray: Preprocessed audio
        """
        try:
            # Load audio file
            audio, orig_sr = sf.read(audio_path, always_2d=False)

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if orig_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

            # Apply DC notch filter if enabled
            if self.dc_filter:
                audio = self._apply_dc_notch_filter(audio)

            # Trim silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

            # Trim to max duration if longer
            if len(audio_trimmed) > self.max_samples:
                audio = audio_trimmed[:self.max_samples]
            else:
                audio = audio_trimmed

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio

        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    def extract_mel_spectrogram(self, audio):
        """
        Extract mel-spectrogram using MLTK approach.

        Args:
            audio: Audio signal (float32)

        Returns:
            Mel-spectrogram (shape: [n_mels, n_time_frames])
        """
        # Apply pre-emphasis
        audio = self._apply_preemphasis(audio)

        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.window_step_length,
            win_length=self.window_size_length,
            window='hann',
            center=True,
            pad_mode='constant'
        )

        # Power spectrogram
        power_spec = np.abs(stft) ** 2

        # Mel filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=False  # Use Slaney-style mel
        )

        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)

        # Apply noise reduction if enabled
        if self.noise_reduction:
            mel_spec = self._apply_noise_reduction(mel_spec)

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        # Return in MLTK format: (mel_bins, time_frames) = (64, T)
        return mel_spec_db

    def normalize_spectrogram(self, mel_spec):
        """
        Normalize mel-spectrogram using training stats.

        Args:
            mel_spec: Mel-spectrogram

        Returns:
            Normalized spectrogram
        """
        return (mel_spec - self.mean) / (self.std + 1e-8)

    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline: load -> extract features -> normalize.

        Args:
            audio_path (str): Path to audio file

        Returns:
            tf.Tensor: Preprocessed features (1, 1, 64, T)
        """
        # Load audio
        audio = self.load_and_preprocess_audio(audio_path)

        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)

        # Normalize
        mel_spec_norm = self.normalize_spectrogram(mel_spec)

        # Add batch and channel dimensions: (64, T) -> (1, 1, 64, T)
        features = mel_spec_norm[np.newaxis, np.newaxis, :, :]

        return tf.constant(features, dtype=tf.float32)

    def predict(self, audio_path, return_probs=False):
        """
        Predict class for an audio file.

        Args:
            audio_path (str): Path to audio file
            return_probs (bool): If True, return probabilities instead of class index

        Returns:
            int or np.ndarray: Predicted class index or probabilities
        """
        # Preprocess
        features = self.preprocess(audio_path)

        # Inference
        logits = self.model(features, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        if return_probs:
            return probs
        else:
            return np.argmax(probs)

    def predict_with_details(self, audio_path):
        """
        Predict with full details (class name, confidence, all probabilities).

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Prediction results with details
        """
        probs = self.predict(audio_path, return_probs=True)
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]

        return {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
        }


def test_inference(model_path, test_audio_path):
    """
    Test inference on a single audio file.

    Args:
        model_path (str): Path to model weights
        test_audio_path (str): Path to test audio file
    """
    print("\n" + "="*60)
    print("SILICON-8 INFERENCE TEST (MLTK Preprocessing)")
    print("="*60)

    # Initialize inference engine
    inference = AudioInference(model_path)

    # Run prediction
    print(f"\nTesting on: {test_audio_path}")
    result = inference.predict_with_details(test_audio_path)

    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Predicted Class: {result['predicted_label']} (index: {result['predicted_class']})")
    print(f"Confidence:      {result['confidence']:.2%}")
    print(f"\nAll Probabilities:")
    for label, prob in result['probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"  {label:12s}: {prob:.2%} {bar}")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Example usage
    MODEL_PATH = "models/best_model.weights.h5"
    TEST_AUDIO = "test_audio/sample.wav"  # Replace with actual test file

    if len(sys.argv) > 1:
        TEST_AUDIO = sys.argv[1]

    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_AUDIO):
        test_inference(MODEL_PATH, TEST_AUDIO)
    else:
        print("Usage: python inference.py <test_audio_path>")
        print(f"Model path: {MODEL_PATH}")
        print(f"Test audio: {TEST_AUDIO}")
