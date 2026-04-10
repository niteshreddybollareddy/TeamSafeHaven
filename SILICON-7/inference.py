"""
SILICON-7 Inference Script for xg26 SoC Deployment

This script provides real-time inference for the 4-class audio classifier:
  - Class 0: Baby Cry
  - Class 1: Dog
  - Class 2: Unknown
  - Class 3: Unusual

Supports variable-length audio (0.13s to 20s) without padding requirements.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
import scipy.signal
from pathlib import Path

# Import model and config
import config
from Model import ImprovedTransformerClassifier


class AudioInference:
    """
    Real-time inference engine for SILICON-7 audio classifier.

    Features:
      - Variable-length audio support (no fixed duration required)
      - Hardware-compatible DSP (same as training pipeline)
      - Efficient mel-spectrogram extraction
      - Automatic normalization using training stats
    """

    def __init__(self, model_path, stats_path=None):
        """
        Initialize inference engine.

        Args:
            model_path (str): Path to saved model weights (.weights.h5)
            stats_path (str): Path to normalization stats (.npy).
                            If None, looks for 'stats.npy' in model directory.
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

        # Class names
        self.class_names = config.CLASS_NAMES

        print(f"\nInference engine ready!")
        print(f"Classes: {self.class_names}")

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
    # DSP FUNCTIONS (same as training preprocessing)
    # =========================================================================

    def hz_to_mel(self, frequencies):
        """Convert Hz to mel scale"""
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    def mel_to_hz(self, mels):
        """Convert mel scale to Hz"""
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    def create_mel_filterbank(self, sr, n_fft, n_mels, fmin, fmax):
        """
        Create mel filterbank matrix.

        Args:
            sr (int): Sample rate
            n_fft (int): FFT size
            n_mels (int): Number of mel bands
            fmin (float): Minimum frequency
            fmax (float): Maximum frequency

        Returns:
            np.ndarray: Mel filterbank matrix (n_mels, n_fft//2 + 1)
        """
        fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
        mel_min = self.hz_to_mel(fmin)
        mel_max = self.hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filters = np.zeros((n_mels, int(1 + n_fft // 2)))

        for i in range(n_mels):
            b_left = bin_points[i]
            b_center = bin_points[i + 1]
            b_right = bin_points[i + 2]

            for k in range(b_left, b_center):
                filters[i, k] = (k - b_left) / (b_center - b_left)
            for k in range(b_center, b_right):
                filters[i, k] = (b_right - k) / (b_right - b_center)

        return filters

    def compute_stft_manual(self, y, n_fft, hop_length):
        """
        Compute STFT manually (hardware-compatible).

        Args:
            y (np.ndarray): Audio signal
            n_fft (int): FFT size
            hop_length (int): Hop length

        Returns:
            np.ndarray: STFT matrix
        """
        pad_amount = n_fft // 2
        y_padded = np.pad(y, (pad_amount, pad_amount), mode='constant')
        n_frames = 1 + (len(y_padded) - n_fft) // hop_length
        window = np.hanning(n_fft)
        stft_matrix = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            frame = y_padded[start:end]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            windowed_frame = frame * window
            fft_frame = np.fft.rfft(windowed_frame, n=n_fft)
            stft_matrix.append(fft_frame)

        return np.array(stft_matrix).T

    def power_to_db_manual(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        """
        Convert power spectrogram to dB scale.

        Args:
            S (np.ndarray): Power spectrogram
            ref (float): Reference power
            amin (float): Minimum amplitude
            top_db (float): Maximum dB range

        Returns:
            np.ndarray: dB-scaled spectrogram
        """
        S_log = 10.0 * np.log10(np.maximum(amin, S))
        S_log -= 10.0 * np.log10(np.maximum(amin, ref))
        if top_db is not None:
            S_log = np.maximum(S_log, S_log.max() - top_db)
        return S_log

    def apply_preemphasis(self, y, coeff=0.97):
        """
        Apply pre-emphasis filter (boosts high frequencies).

        Args:
            y (np.ndarray): Audio signal
            coeff (float): Pre-emphasis coefficient

        Returns:
            np.ndarray: Pre-emphasized signal
        """
        return np.append(y[0], y[1:] - coeff * y[:-1])

    def load_and_preprocess_audio(self, audio_path, target_sr=config.SAMPLE_RATE):
        """
        Load and preprocess audio file (same as training pipeline).

        Args:
            audio_path (str): Path to audio file
            target_sr (int): Target sample rate (default: 16000)

        Returns:
            tuple: (preprocessed_audio, sample_rate)
        """
        try:
            # 1. Load audio with SoundFile
            y, orig_sr = sf.read(audio_path, always_2d=False)

            # 2. Handle multi-channel (mix to mono)
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # 3. Resample if needed
            if orig_sr != target_sr:
                gcd = np.gcd(orig_sr, target_sr)
                up = target_sr // gcd
                down = orig_sr // gcd
                y = scipy.signal.resample_poly(y, up, down)

            # 4. Pre-emphasis
            y = self.apply_preemphasis(y)

            # 5. Silence trimming (less aggressive)
            frame_len = 2048
            hop = 512
            y_sq = y**2

            energy = np.array([np.sum(y_sq[i:i+frame_len]) for i in range(0, len(y), hop)])

            if len(energy) > 0:
                max_energy = np.max(energy)
                if max_energy > 0:
                    threshold = max_energy * (10**(-40/10))
                    active_frames = np.where(energy > threshold)[0]

                    if len(active_frames) > 0:
                        start = active_frames[0] * hop
                        end = min(len(y), (active_frames[-1] * hop) + frame_len)
                        start = max(0, start - hop*2)
                        end = min(len(y), end + hop*2)
                        y = y[start:end]

            # 6. Peak normalization
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # 7. Adaptive duration handling (max 8 seconds)
            MAX_DURATION = 8.0
            max_samples = int(MAX_DURATION * target_sr)

            if len(y) > max_samples:
                # Audio is longer than 8 seconds - needs special handling
                # This will be handled in extract_melspectrogram_with_windowing
                pass

            return y, target_sr

        except Exception as e:
            raise RuntimeError(f"Error loading audio {audio_path}: {e}")

    def extract_melspectrogram(self, audio_path):
        """
        Extract mel-spectrogram from audio file.

        Args:
            audio_path (str): Path to audio file

        Returns:
            np.ndarray: Mel-spectrogram (64, T) where T varies
        """
        # Load and preprocess
        y, sr = self.load_and_preprocess_audio(audio_path)

        if len(y) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")

        # Compute STFT
        stft_complex = self.compute_stft_manual(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        power_spec = np.abs(stft_complex)**2

        # Apply mel filterbank
        mel_basis = self.create_mel_filterbank(sr, config.N_FFT, config.N_MELS, fmin=50, fmax=8000)
        mel_spec = np.dot(mel_basis, power_spec)

        # Convert to dB
        mel_spec_db = self.power_to_db_manual(mel_spec, ref=np.max(mel_spec))

        return mel_spec_db

    def preprocess_audio(self, audio_path):
        """
        Full preprocessing pipeline for inference.

        Args:
            audio_path (str): Path to audio file

        Returns:
            tf.Tensor: Preprocessed tensor ready for model (1, 1, 64, T)
        """
        # Extract mel-spectrogram
        mel_spec = self.extract_melspectrogram(audio_path)

        # Normalize using training stats
        mel_spec = (mel_spec - self.mean) / (self.std + 1e-8)

        # Add batch and channel dimensions: (64, T) -> (1, 1, 64, T)
        mel_spec = mel_spec[np.newaxis, np.newaxis, ...]
        mel_spec = mel_spec.astype(np.float32)

        return tf.constant(mel_spec)

    def find_most_energetic_window(self, y, sr, window_duration=8.0):
        """
        Find the most energetic 8-second window in the audio.

        Args:
            y: Audio signal
            sr: Sample rate
            window_duration: Window duration in seconds

        Returns:
            Audio segment of window_duration
        """
        window_samples = int(window_duration * sr)

        if len(y) <= window_samples:
            return y

        # Calculate RMS energy for each possible window
        best_energy = -1
        best_start = 0

        for start in range(0, len(y) - window_samples + 1, sr // 2):  # Step by 0.5 seconds
            window = y[start:start + window_samples]
            energy = np.sqrt(np.mean(window ** 2))

            if energy > best_energy:
                best_energy = energy
                best_start = start

        return y[best_start:best_start + window_samples]

    def extract_melspectrogram_from_array(self, y, sr):
        """
        Extract mel-spectrogram from audio array (not file).

        Args:
            y: Audio signal array
            sr: Sample rate

        Returns:
            np.ndarray: Mel-spectrogram
        """
        # Apply pre-emphasis if not already done
        # (This assumes y is already pre-emphasized from load_and_preprocess_audio)

        # Compute STFT
        stft_complex = self.compute_stft_manual(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        power_spec = np.abs(stft_complex)**2

        # Apply mel filterbank
        mel_basis = self.create_mel_filterbank(sr, config.N_FFT, config.N_MELS, fmin=50, fmax=8000)
        mel_spec = np.dot(mel_basis, power_spec)

        # Convert to dB
        mel_spec_db = self.power_to_db_manual(mel_spec, ref=np.max(mel_spec))

        return mel_spec_db

    def predict_with_windowing(self, audio_path):
        """
        Predict with adaptive windowing based on audio duration.

        Strategy:
        - If audio ≤ 8s: Use as-is
        - If 8s < audio ≤ 12s: Take most energetic 8s window
        - If audio > 12s: Use sliding window voting (3 windows)

        Returns:
            dict: Prediction results with averaged probabilities
        """
        MAX_DURATION = 8.0
        MEDIUM_THRESHOLD = 9.0  # Lower threshold to use voting for 10s files

        # Load and preprocess audio
        y, sr = self.load_and_preprocess_audio(audio_path)
        duration = len(y) / sr
        max_samples = int(MAX_DURATION * sr)

        print(f"Audio duration: {duration:.2f}s")

        # Case 1: Audio ≤ 8s - use as-is
        if duration <= MAX_DURATION:
            print("Using full audio (≤8s)")
            mel_spec = self.extract_melspectrogram_from_array(y, sr)
            mel_spec_norm = (mel_spec - self.mean) / (self.std + 1e-8)
            mel_spec_norm = mel_spec_norm[np.newaxis, np.newaxis, ...].astype(np.float32)

            logits = self.model(tf.constant(mel_spec_norm), training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

            return self._format_result(probs, windows_analyzed=1)

        # Case 2: 8s < audio ≤ 12s - take most energetic 8s window
        elif duration <= MEDIUM_THRESHOLD:
            print(f"Taking most energetic 8s window from {duration:.2f}s audio")
            y_windowed = self.find_most_energetic_window(y, sr, MAX_DURATION)

            mel_spec = self.extract_melspectrogram_from_array(y_windowed, sr)
            mel_spec_norm = (mel_spec - self.mean) / (self.std + 1e-8)
            mel_spec_norm = mel_spec_norm[np.newaxis, np.newaxis, ...].astype(np.float32)

            logits = self.model(tf.constant(mel_spec_norm), training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

            return self._format_result(probs, windows_analyzed=1)

        # Case 3: audio > 12s - sliding window voting (3 windows)
        else:
            print(f"Using sliding window voting on {duration:.2f}s audio")

            # Extract 3 windows: [0-8s], [middle 8s], [last 8s]
            window1 = y[:max_samples]  # First 8s

            mid_start = (len(y) - max_samples) // 2
            window2 = y[mid_start:mid_start + max_samples]  # Middle 8s

            window3 = y[-max_samples:]  # Last 8s

            # Run inference on each window
            all_probs = []
            for i, window in enumerate([window1, window2, window3], 1):
                mel_spec = self.extract_melspectrogram_from_array(window, sr)
                mel_spec_norm = (mel_spec - self.mean) / (self.std + 1e-8)
                mel_spec_norm = mel_spec_norm[np.newaxis, np.newaxis, ...].astype(np.float32)

                logits = self.model(tf.constant(mel_spec_norm), training=False)
                probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
                all_probs.append(probs)

                print(f"  Window {i}: {self.class_names[np.argmax(probs)]} ({probs[np.argmax(probs)]*100:.1f}%)")

            # Average probabilities across all windows
            avg_probs = np.mean(all_probs, axis=0)

            return self._format_result(avg_probs, windows_analyzed=3)

    def _format_result(self, probs, windows_analyzed=1):
        """Format prediction result."""
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
            },
            'windows_analyzed': windows_analyzed
        }

    def predict(self, audio_path, return_probs=True):
        """
        Run inference on audio file with adaptive windowing.

        Args:
            audio_path (str): Path to audio file
            return_probs (bool): If True, return probabilities; if False, return logits

        Returns:
            dict: Prediction results with keys:
                - 'class_idx': Predicted class index (0-3)
                - 'class_name': Predicted class name
                - 'confidence': Confidence score (0-1)
                - 'probabilities': All class probabilities
                - 'windows_analyzed': Number of windows analyzed
        """
        # Use windowing-based prediction for better handling of long audio
        return self.predict_with_windowing(audio_path)

    def predict_batch(self, audio_paths):
        """
        Run inference on multiple audio files.

        Note: For variable-length audio, batch processing requires padding
        to the longest sample in the batch. For most efficient processing,
        use predict() for single samples.

        Args:
            audio_paths (list): List of audio file paths

        Returns:
            list: List of prediction results
        """
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
    """Example usage of inference engine"""
    import argparse

    parser = argparse.ArgumentParser(description='SILICON-7 Audio Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, default='models/best_model.weights.h5',
                       help='Path to model weights')
    parser.add_argument('--stats', type=str, default=None,
                       help='Path to stats file (default: models/stats.npy)')

    args = parser.parse_args()

    # Initialize inference engine
    print("="*60)
    print("SILICON-7 Audio Classification Inference")
    print("="*60)

    inference = AudioInference(args.model, args.stats)

    # Handle single file or directory
    audio_path = Path(args.audio)

    if audio_path.is_file():
        # Single file inference
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
        # Directory inference
        audio_files = list(audio_path.glob('*.wav')) + list(audio_path.glob('*.ogg'))

        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return

        print(f"\nProcessing {len(audio_files)} files from: {audio_path}")

        results = inference.predict_batch([str(f) for f in audio_files])

        # Summary
        print("\n" + "="*60)
        print("BATCH PREDICTION RESULTS")
        print("="*60)

        for audio_file, result in zip(audio_files, results):
            if result is not None:
                print(f"{audio_file.name:40s} → {result['class_name']:12s} ({result['confidence']*100:5.2f}%)")
            else:
                print(f"{audio_file.name:40s} → ERROR")

        # Class distribution
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


"""
================================================================================
INFERENCE USAGE EXAMPLES:
================================================================================

1. SINGLE FILE INFERENCE:
   python inference.py --audio test_audio/baby_cry.wav

2. BATCH INFERENCE (DIRECTORY):
   python inference.py --audio test_audio/ --model models/best_model.weights.h5

3. CUSTOM MODEL AND STATS:
   python inference.py --audio audio.wav --model path/to/model.weights.h5 --stats path/to/stats.npy

4. PYTHON API USAGE:
   ```python
   from inference import AudioInference

   # Initialize
   inference = AudioInference('models/best_model.weights.h5')

   # Single prediction
   result = inference.predict('audio.wav')
   print(f"Class: {result['class_name']}, Confidence: {result['confidence']:.2f}")

   # Batch prediction
   results = inference.predict_batch(['audio1.wav', 'audio2.wav', 'audio3.wav'])
   ```

================================================================================
DEPLOYMENT ON xg26 SoC:
================================================================================

For embedded deployment:
  1. Audio must be 16kHz mono (resample if needed)
  2. Duration can be 0.13s to 20s (no fixed length required!)
  3. Model handles variable lengths natively (no padding)
  4. Memory usage: ~15KB for model parameters + variable for audio buffer
  5. Latency: ~10-50ms depending on audio length and hardware

For C/C++ deployment:
  1. Port DSP functions (mel-spectrogram extraction)
  2. Load model using TensorFlow Lite or ONNX runtime
  3. Use saved normalization stats (mean, std)
  4. Forward pass through model
  5. Apply softmax to get probabilities

================================================================================
"""
