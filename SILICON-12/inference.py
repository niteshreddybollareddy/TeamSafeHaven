"""
SILICON-12: Inference Script for 3-Class Audio Classification
===============================================================
Baby Cry, Dog Barking, Unknown classification using CryNet3Class.

CRITICAL DESIGN DECISIONS (matching SILICON-9's proven approach):
1. Single max-energy window (NOT multi-window averaging)
2. EXACT same preprocessing as training
3. Global z-score normalization using training stats
4. center=False in STFT for predictable frame count

WHY SINGLE WINDOW (not sliding windows):
- SILICON-10's multi-window averaging DILUTED the signal
- A clear cry in one window gets averaged with silence in others
- Max-energy window captures the loudest/most important part
- Training uses max-energy window, inference must match

Usage:
    # Single file inference
    python inference.py --audio path/to/audio.wav

    # Batch inference on directory
    python inference.py --dir path/to/audio_folder

    # Specify custom checkpoint
    python inference.py --audio test.wav --checkpoint experiments/run_xxx/checkpoints
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import scipy.signal
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import CryNet3Class, count_parameters


class AudioClassifierInference:
    """
    Inference engine for SILICON-12 3-Class Audio Classifier.

    Key features:
    - Single max-energy window (matches training)
    - EXACT same preprocessing pipeline as training
    - Global z-score normalization with training stats
    - 3-class output: Baby Cry, Dog Barking, Unknown
    """

    def __init__(self, checkpoint_dir=None, verbose=True):
        """
        Initialize the inference engine.

        Args:
            checkpoint_dir: Path to checkpoint directory containing:
                           - best_model.weights.h5
                           - stats.npy
                           If None, searches default locations
            verbose: Print debug information
        """
        self.verbose = verbose

        # Audio settings from config
        self.sample_rate = config.SAMPLE_RATE_HZ
        self.sample_length = config.SAMPLE_LENGTH
        self.sample_length_ms = config.SAMPLE_LENGTH_MS

        # Spectrogram settings
        self.window_size_length = config.WINDOW_SIZE_LENGTH
        self.window_step_length = config.WINDOW_STEP_LENGTH
        self.n_fft = config.FFT_LENGTH
        self.n_mels = config.FILTERBANK_N_CHANNELS
        self.fmin = config.FILTERBANK_LOWER_BAND_LIMIT
        self.fmax = config.FILTERBANK_UPPER_BAND_LIMIT
        self.n_time_frames = config.N_TIME_FRAMES

        # Class names
        self.class_names = config.CLASS_NAMES
        self.num_classes = config.NUM_CLASSES

        # Find checkpoint directory
        self.checkpoint_dir = self._find_checkpoint_dir(checkpoint_dir)

        # Load normalization stats
        self.stats = self._load_stats()

        # Load model
        self.model = self._load_model()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SILICON-12 INFERENCE ENGINE INITIALIZED")
            print(f"{'='*60}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Window: {self.sample_length_ms} ms ({self.sample_length} samples)")
            print(f"  Spectrogram: ({self.n_mels}, {self.n_time_frames})")
            print(f"  Classes: {self.class_names}")
            print(f"  Normalization: mean={self.stats['mean']:.4f}, std={self.stats['std']:.4f}")
            print(f"  Strategy: Single max-energy window (matches training)")
            print(f"{'='*60}\n")

    def _find_checkpoint_dir(self, checkpoint_dir):
        """Find checkpoint directory."""
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            return checkpoint_dir

        # Search in default locations
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, 'models'),
            os.path.join(base_dir, 'checkpoints'),
        ]

        # Also search in experiments directory for latest run
        exp_dir = os.path.join(base_dir, 'experiments')
        if os.path.exists(exp_dir):
            exp_runs = sorted([d for d in os.listdir(exp_dir)
                              if d.startswith('crynet3class_')])
            if exp_runs:
                latest = exp_runs[-1]
                candidates.insert(0, os.path.join(exp_dir, latest, 'checkpoints'))

        for candidate in candidates:
            weights_path = os.path.join(candidate, 'best_model.weights.h5')
            if os.path.exists(weights_path):
                if self.verbose:
                    print(f"Found checkpoint: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"No checkpoint found. Please train a model first or specify --checkpoint.\n"
            f"Searched: {candidates}"
        )

    def _load_stats(self):
        """Load normalization statistics from training."""
        stats_path = os.path.join(self.checkpoint_dir, 'stats.npy')

        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True).item()
            if self.verbose:
                print(f"Loaded stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            return stats
        else:
            print(f"WARNING: stats.npy not found at {stats_path}")
            print(f"Using default normalization (may affect accuracy)")
            return {'mean': 0.0, 'std': 1.0}

    def _load_model(self):
        """Load the trained model."""
        weights_path = os.path.join(self.checkpoint_dir, 'best_model.weights.h5')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        if self.verbose:
            print(f"Loading model: {weights_path}")

        # Build model architecture
        model = CryNet3Class(
            num_classes=self.num_classes,
            dropout=config.DROPOUT
        )

        # Build model with dummy input
        dummy_input = tf.zeros((1, 1, self.n_mels, self.n_time_frames))
        model(dummy_input, training=False)

        # Load weights
        model.load_weights(weights_path)

        params = count_parameters(model)
        if self.verbose:
            print(f"Model loaded: {params:,} parameters")

        return model

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """Apply DC notch filter to remove DC offset."""
        b = [1, -1]
        a = [1, -coefficient]
        return scipy.signal.lfilter(b, a, audio)

    def _apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def _select_max_energy_window(self, audio, target_length, debug=False):
        """
        Select window with maximum energy (SILICON-9's key technique).

        This ensures the loudest/most important part of audio is captured.
        MUST match training preprocessing exactly.

        Args:
            audio: Raw audio signal
            target_length: Window length in samples
            debug: Print debug info

        Returns:
            Audio window with maximum energy
        """
        if len(audio) <= target_length:
            # Pad if too short
            pad_length = target_length - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')

        # Compute RMS energy
        frame_energy = librosa.feature.rms(
            y=audio,
            frame_length=config.ENERGY_WINDOW_FRAME_LENGTH,
            hop_length=config.ENERGY_WINDOW_HOP_LENGTH
        ).squeeze()

        # Find max energy frame
        max_energy_frame = np.argmax(frame_energy)
        center_sample = max_energy_frame * config.ENERGY_WINDOW_HOP_LENGTH

        # Safe clamping
        start_idx = center_sample - target_length // 2
        start_idx = max(0, min(start_idx, len(audio) - target_length))
        end_idx = start_idx + target_length

        if debug:
            print(f"    Max energy at frame {max_energy_frame} (sample {center_sample})")
            print(f"    Window: [{start_idx}:{end_idx}]")

        return audio[start_idx:end_idx]

    def load_and_preprocess_audio(self, audio_path, debug=False):
        """
        Load and preprocess audio file.

        MUST match training preprocessing exactly:
        1. Load and resample
        2. DC notch filter
        3. Max-energy window selection (BEFORE silence trim)
        4. Light silence trim
        5. Peak normalization

        Args:
            audio_path: Path to audio file
            debug: Print debug info

        Returns:
            Preprocessed audio array (exactly sample_length samples)
        """
        # Load audio
        audio, orig_sr = sf.read(audio_path, always_2d=False)
        if debug:
            print(f"  [1] Loaded: {len(audio)} samples @ {orig_sr}Hz")

        # Stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            if debug:
                print(f"  [2] Stereo->Mono")

        # Resample if needed
        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
            if debug:
                print(f"  [3] Resampled to {self.sample_rate}Hz: {len(audio)} samples")

        # DC notch filter
        if config.DC_NOTCH_FILTER_ENABLE:
            audio = self._apply_dc_notch_filter(audio)
            if debug:
                print(f"  [4] DC filter applied")

        # Max-energy window selection (BEFORE silence trim - CRITICAL)
        if len(audio) < self.sample_length:
            pad_length = self.sample_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
            if debug:
                print(f"  [5] Padded to {self.sample_length} samples")
        else:
            audio = self._select_max_energy_window(audio, self.sample_length, debug)
            if debug:
                print(f"  [5] Max-energy window selected")

        # Light silence trim AFTER energy windowing
        trim_db = config.SILENCE_TRIM_TOP_DB
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=trim_db)

        # Re-pad if trimming made it too short
        if len(audio_trimmed) < self.sample_length:
            pad_length = self.sample_length - len(audio_trimmed)
            audio = np.pad(audio_trimmed, (0, pad_length), mode='constant')
        else:
            audio = audio_trimmed[:self.sample_length]

        if debug:
            print(f"  [6] Trimmed (top_db={trim_db}): {len(audio)} samples")

        # Peak normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            if debug:
                print(f"  [7] Normalized: [{audio.min():.3f}, {audio.max():.3f}]")

        return audio

    def extract_spectrogram(self, audio, debug=False):
        """
        Extract mel-spectrogram from audio.

        MUST match training exactly:
        - Pre-emphasis
        - STFT with center=False
        - Mel filterbank
        - Log scale (dB)
        - Z-score normalization with training stats

        Args:
            audio: Audio array (float32, sample_length samples)
            debug: Print debug info

        Returns:
            Normalized mel-spectrogram (shape: [n_mels, n_time_frames])
        """
        if debug:
            print(f"  [8] Audio: {len(audio)} samples")

        # Pre-emphasis
        audio = self._apply_preemphasis(audio)

        # STFT with center=False (CRITICAL for predictable frame count)
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.window_step_length,
            win_length=self.window_size_length,
            window='hann',
            center=False  # CRITICAL: matches training
        )

        if debug:
            print(f"  [9] STFT: {stft.shape}")

        # Power spectrogram
        power_spec = np.abs(stft) ** 2

        # Mel filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=False
        )

        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)

        if debug:
            print(f"  [10] Mel: {mel_spec.shape}")

        # Noise reduction (if enabled in training)
        if config.NOISE_REDUCTION_ENABLE:
            noise_floor = np.median(mel_spec[:, :5], axis=1, keepdims=True)
            mel_spec_clean = mel_spec - noise_floor
            mel_spec = np.maximum(
                mel_spec_clean,
                mel_spec * config.NOISE_REDUCTION_MIN_SIGNAL_REMAINING
            )

        # Log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        if debug:
            print(f"  [11] Log dB: [{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}]")

        # Z-score normalization using training stats
        mel_spec_norm = (mel_spec_db - self.stats['mean']) / (self.stats['std'] + 1e-8)

        if debug:
            print(f"  [12] Normalized: {mel_spec_norm.shape}")

        return mel_spec_norm.astype(np.float32)

    def preprocess_for_model(self, audio, debug=False):
        """
        Full preprocessing pipeline for model input.

        Args:
            audio: Raw audio array
            debug: Print debug info

        Returns:
            Model input tensor (shape: [1, 1, n_mels, n_time_frames])
        """
        mel_spec = self.extract_spectrogram(audio, debug)

        # Ensure exact frame count
        if mel_spec.shape[1] > self.n_time_frames:
            mel_spec = mel_spec[:, :self.n_time_frames]
        elif mel_spec.shape[1] < self.n_time_frames:
            pad_width = self.n_time_frames - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

        # Add batch and channel dimensions: (n_mels, n_frames) -> (1, 1, n_mels, n_frames)
        mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]

        return mel_spec

    def predict(self, audio_path, debug=False):
        """
        Run inference on an audio file.

        Uses single max-energy window (NOT multi-window averaging).

        Args:
            audio_path: Path to audio file
            debug: Print detailed debug info

        Returns:
            dict with prediction results
        """
        if debug or self.verbose:
            print(f"\nProcessing: {audio_path}")

        start_time = time.perf_counter()

        # Load and preprocess audio
        audio = self.load_and_preprocess_audio(audio_path, debug=debug)

        # Prepare for model
        input_tensor = self.preprocess_for_model(audio, debug=debug)

        # Run inference
        probabilities = self.model(input_tensor, training=False).numpy()[0]

        inference_time = (time.perf_counter() - start_time) * 1000

        # Get prediction
        class_id = np.argmax(probabilities)
        confidence = probabilities[class_id]
        class_name = self.class_names[class_id]

        return {
            'class_id': int(class_id),
            'class_name': class_name,
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'inference_time_ms': float(inference_time),
            'file': os.path.basename(audio_path)
        }

    def predict_batch(self, audio_dir, extensions=['.wav', '.mp3', '.flac', '.ogg']):
        """
        Run inference on all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files
            extensions: List of audio file extensions to process

        Returns:
            List of prediction results
        """
        audio_dir = Path(audio_dir)
        results = []

        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f'**/*{ext}'))

        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return results

        print(f"\nProcessing {len(audio_files)} files...")

        for i, audio_path in enumerate(audio_files):
            try:
                result = self.predict(str(audio_path))
                results.append(result)

                if self.verbose:
                    prob_str = " | ".join([f"{cn}: {result['probabilities'][cn]*100:.1f}%"
                                           for cn in self.class_names])
                    print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}: "
                          f"{result['class_name']} ({result['confidence']*100:.1f}%)")
            except Exception as e:
                print(f"  [{i+1}/{len(audio_files)}] ERROR: {audio_path.name}: {e}")

        return results


def print_result(result):
    """Pretty print a prediction result."""
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULT")
    print(f"{'='*60}")

    if 'file' in result:
        print(f"  File: {result['file']}")

    print(f"  Inference time: {result['inference_time_ms']:.2f} ms")

    print(f"\n  Predicted: {result['class_name'].upper()}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")

    print(f"\n  All probabilities:")
    for class_name, prob in result['probabilities'].items():
        bar_length = int(prob * 30)
        bar = '=' * bar_length + ' ' * (30 - bar_length)
        marker = " <--" if class_name == result['class_name'] else ""
        print(f"    {class_name:12s} [{bar}] {prob*100:5.1f}%{marker}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SILICON-12 3-Class Audio Classifier Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --audio test.wav
  python inference.py --dir audio_folder/
  python inference.py --audio test.wav --checkpoint experiments/run_xxx/checkpoints

Key Features:
  - Single max-energy window (matches training, no averaging)
  - 3 classes: Baby Cry, Dog Barking, Unknown
  - Fast inference (~15ms per file)
        """
    )

    parser.add_argument('--audio', type=str, help='Path to audio file for inference')
    parser.add_argument('--dir', type=str, help='Directory of audio files for batch inference')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug info')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.audio, args.dir]):
        parser.print_help()
        print("\nError: Please specify --audio or --dir")
        sys.exit(1)

    # Initialize inference engine
    engine = AudioClassifierInference(
        checkpoint_dir=args.checkpoint,
        verbose=not args.quiet
    )

    # Run inference
    if args.audio:
        result = engine.predict(args.audio, debug=args.debug)
        print_result(result)

    elif args.dir:
        results = engine.predict_batch(args.dir)

        if results:
            print(f"\n{'='*60}")
            print(f"BATCH SUMMARY")
            print(f"{'='*60}")
            print(f"  Total files: {len(results)}")

            # Count by class
            class_counts = {}
            for r in results:
                cls = r['class_name']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            print(f"\n  Predictions:")
            for cls, count in sorted(class_counts.items()):
                print(f"    {cls}: {count} ({count/len(results)*100:.1f}%)")

            # Average confidence
            avg_conf = np.mean([r['confidence'] for r in results])
            print(f"\n  Average confidence: {avg_conf*100:.1f}%")

            # Average inference time
            avg_time = np.mean([r['inference_time_ms'] for r in results])
            print(f"  Average inference time: {avg_time:.2f} ms")
            print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
