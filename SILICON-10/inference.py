"""
SILICON-10: Inference Script for Baby Cry Detection
====================================================
Real-time and batch inference using trained CNN model.

PROFESSOR FIXES IMPLEMENTED:
  1. Energy threshold for window filtering (domain alignment)
  2. RMS normalization instead of peak normalization
  3. Max/percentile aggregation instead of mean
  4. Confidence calibration and logging

Usage:
    # Single file inference
    python inference.py --audio path/to/audio.wav

    # Batch inference on directory
    python inference.py --dir path/to/audio_folder

    # Real-time microphone inference
    python inference.py --realtime

    # Specify custom model path
    python inference.py --audio test.wav --model models/best_model.keras
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
from model_cnn import BabyCryDetectorCNN


# ============================================================================
# PROFESSOR FIX 1 & 4: Energy threshold and confidence calibration constants
# ============================================================================
ENERGY_THRESHOLD = 0.01          # Minimum RMS energy to process window (skip silence)
RMS_FLOOR = 1e-4                 # Minimum RMS for normalization (avoid boosting silence)
CONFIDENCE_THRESHOLD = 0.65      # Empirical threshold for baby cry detection
AGGREGATION_PERCENTILE = 90      # Use 90th percentile for multi-window aggregation
AGGREGATION_MODE = "percentile"  # "percentile" (safer for noise) or "max" (better for short cries)


class BabyCryInference:
    """
    Inference engine for SILICON-10 Baby Cry Detection model.

    Handles:
    - Audio loading and preprocessing (with energy filtering)
    - Mel-spectrogram extraction (matching training pipeline)
    - Model inference with confidence scores
    - Multi-window aggregation with max/percentile
    - Real-time streaming inference
    """

    def __init__(self, model_path=None, verbose=True):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to trained model (.keras or .h5 file)
                       If None, looks for best_model.keras in models/
            verbose: Print debug information
        """
        self.verbose = verbose
        self.sample_rate = config.SAMPLE_RATE_HZ
        self.sample_length = config.SAMPLE_LENGTH
        self.sample_length_ms = config.SAMPLE_LENGTH_MS

        # Spectrogram settings
        self.window_size_ms = config.WINDOW_SIZE_MS
        self.window_step_ms = config.WINDOW_STEP_MS
        self.n_fft = config.FFT_LENGTH
        self.n_mels = config.FILTERBANK_N_CHANNELS
        self.fmin = config.FILTERBANK_LOWER_BAND_LIMIT
        self.fmax = config.FILTERBANK_UPPER_BAND_LIMIT

        # Calculate frame parameters
        self.window_size_length = int((self.window_size_ms * self.sample_rate) / 1000)
        self.window_step_length = int((self.window_step_ms * self.sample_rate) / 1000)

        # Class names
        self.class_names = config.CLASS_NAMES
        self.num_classes = config.NUM_CLASSES

        # Load normalization stats (mean/std from training)
        self.stats = self._load_stats()

        # Load model
        self.model = self._load_model(model_path)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SILICON-10 INFERENCE ENGINE INITIALIZED")
            print(f"{'='*60}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Window length: {self.sample_length_ms} ms ({self.sample_length} samples)")
            print(f"  Window size: {self.window_size_ms} ms")
            print(f"  Window step: {self.window_step_ms} ms")
            print(f"  Mel bands: {self.n_mels}")
            print(f"  Classes: {self.class_names}")
            print(f"  Normalization: mean={self.stats['mean']:.4f}, std={self.stats['std']:.4f}")
            print(f"\n  [PROF FIX 1] Energy threshold: {ENERGY_THRESHOLD}")
            print(f"  [PROF FIX 2] RMS floor: {RMS_FLOOR}")
            agg_desc = f"{AGGREGATION_PERCENTILE}th percentile" if AGGREGATION_MODE == "percentile" else "max"
            print(f"  [PROF FIX 3] Aggregation: {AGGREGATION_MODE} ({agg_desc})")
            print(f"  [PROF FIX 4] Confidence threshold: {CONFIDENCE_THRESHOLD}")
            print(f"{'='*60}\n")

    def _load_stats(self):
        """Load normalization statistics from training."""
        stats_path = os.path.join(os.path.dirname(__file__), 'models', 'stats.npy')

        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True).item()
            if self.verbose:
                print(f"Loaded normalization stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            return stats
        else:
            # Fallback to default values if stats not found
            print(f"WARNING: stats.npy not found at {stats_path}")
            print(f"Using default normalization (may affect accuracy)")
            return {'mean': 0.0, 'std': 1.0}

    def _load_model(self, model_path):
        """Load the trained model."""
        if model_path is None:
            # Look for best model in default location
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            candidates = [
                os.path.join(model_dir, 'best_model.weights.h5'),
                os.path.join(model_dir, 'best_model.keras'),
                os.path.join(model_dir, 'best_model.h5'),
                os.path.join(model_dir, 'model_final.keras'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"No model found. Please specify --model path or train a model first.\n"
                    f"Searched: {candidates}"
                )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.verbose:
            print(f"Loading model: {model_path}")

        # Build model architecture
        model = BabyCryDetectorCNN(
            num_classes=self.num_classes,
            dropout=config.CNN_DROPOUT,
            use_depthwise=config.USE_DEPTHWISE_SEPARABLE
        )

        # Build model with dummy input to initialize weights
        n_time_frames = config.N_TIME_FRAMES
        dummy_input = tf.zeros((1, 1, self.n_mels, n_time_frames))
        model(dummy_input, training=False)

        # Load weights
        model.load_weights(model_path)

        if self.verbose:
            print(f"Model loaded successfully!")

        return model

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """Apply DC notch filter to remove DC offset."""
        b = [1, -1]
        a = [1, -coefficient]
        return scipy.signal.lfilter(b, a, audio)

    def _apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    # =========================================================================
    # PROFESSOR FIX 1: Check window energy before processing
    # =========================================================================
    def _check_window_energy(self, audio):
        """
        Check if audio window has sufficient energy (not silence).

        PROFESSOR FIX 1: Skip silent windows to align inference with training.
        Training used energy-based windowing, so inference should too.

        Args:
            audio: Audio window

        Returns:
            (has_energy, rms_value): Tuple of boolean and RMS value
        """
        rms = np.sqrt(np.mean(audio ** 2))
        has_energy = rms > ENERGY_THRESHOLD
        return has_energy, rms

    # =========================================================================
    # PROFESSOR FIX 2: RMS normalization instead of peak normalization
    # =========================================================================
    def _normalize_audio_rms(self, audio):
        """
        Normalize audio using RMS with floor.

        PROFESSOR FIX 2: Peak normalization artificially boosts silence.
        RMS normalization preserves relative energy information.

        Args:
            audio: Audio array

        Returns:
            RMS-normalized audio
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > RMS_FLOOR:
            audio = audio / rms
        # If RMS is below floor, don't normalize (preserve silence as silence)
        return audio

    def load_audio(self, audio_path, normalize_per_file=True):
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file
            normalize_per_file: If True, normalize once per file (recommended)

        Returns:
            Preprocessed audio array (correct sample rate, optionally normalized)
        """
        # Load audio
        audio, orig_sr = sf.read(audio_path, always_2d=False)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

        # Apply DC notch filter
        audio = self._apply_dc_notch_filter(audio)

        # Light silence trim (preserves structure)
        audio, _ = librosa.effects.trim(audio, top_db=40)

        # PROFESSOR FIX 2: RMS normalization once per file (not per window)
        if normalize_per_file:
            audio = self._normalize_audio_rms(audio)

        return audio

    def extract_spectrogram(self, audio):
        """
        Extract mel-spectrogram from audio.

        Args:
            audio: Audio array (float32)

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
            htk=False
        )

        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)

        # Convert to log scale (dB) - MUST MATCH TRAINING PIPELINE
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        # Z-score normalization using training stats - MUST MATCH TRAINING PIPELINE
        mel_spec_norm = (mel_spec_db - self.stats['mean']) / (self.stats['std'] + 1e-8)

        return mel_spec_norm.astype(np.float32)

    def _select_max_energy_window(self, audio, target_length):
        """
        Select window with maximum energy (matches training).

        Args:
            audio: Full audio array
            target_length: Window length in samples

        Returns:
            Audio window with highest energy
        """
        if len(audio) <= target_length:
            # Pad if too short
            pad_length = target_length - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')

        # Compute RMS energy
        frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512).squeeze()
        max_energy_frame = np.argmax(frame_energy)
        center_sample = max_energy_frame * 512

        # Extract window centered on max energy
        start_idx = center_sample - target_length // 2
        start_idx = max(0, min(start_idx, len(audio) - target_length))
        end_idx = start_idx + target_length

        return audio[start_idx:end_idx]

    def _extract_sliding_windows(self, audio, hop_ratio=0.5):
        """
        Extract sliding windows from audio with energy filtering.

        PROFESSOR FIX 1: Skip windows below energy threshold.

        Args:
            audio: Full audio array
            hop_ratio: Overlap ratio (0.5 = 50% overlap)

        Returns:
            List of (window, start_idx, rms) tuples for valid windows
        """
        window_length = self.sample_length
        hop_length = int(window_length * hop_ratio)

        windows = []
        total_windows = 0
        start = 0

        while start + window_length <= len(audio):
            total_windows += 1
            window = audio[start:start + window_length]

            # PROFESSOR FIX 1: Check energy before adding
            has_energy, rms = self._check_window_energy(window)

            if has_energy:
                windows.append((window, start, rms))

            start += hop_length

        # If no windows passed energy filter, use max-energy window as fallback
        if not windows and len(audio) >= window_length:
            window = self._select_max_energy_window(audio, window_length)
            _, rms = self._check_window_energy(window)
            windows.append((window, 0, rms))
            if self.verbose:
                print(f"  [Energy Filter] All windows filtered, using max-energy fallback")

        # MINOR SUGGESTION 1: Log how many windows survive filtering
        if self.verbose and total_windows > 0:
            print(f"  [Energy Filter] Windows kept: {len(windows)} / {total_windows} "
                  f"({len(windows)/total_windows*100:.0f}%)")

        return windows

    def preprocess_audio(self, audio):
        """
        Preprocess audio for model input (single window).

        Args:
            audio: Raw audio array (should be exactly sample_length)

        Returns:
            Model input tensor (shape: [1, 1, n_mels, n_time_frames])
        """
        # Ensure correct length
        if len(audio) > self.sample_length:
            audio = audio[:self.sample_length]
        elif len(audio) < self.sample_length:
            audio = np.pad(audio, (0, self.sample_length - len(audio)), mode='constant')

        # Extract spectrogram
        mel_spec = self.extract_spectrogram(audio)

        # Expected time frames
        expected_frames = config.N_TIME_FRAMES
        current_frames = mel_spec.shape[1]

        # Adjust to expected size
        if current_frames > expected_frames:
            mel_spec = mel_spec[:, :expected_frames]
        elif current_frames < expected_frames:
            pad_width = expected_frames - current_frames
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

        # Add batch and channel dimensions: (n_mels, n_frames) -> (1, 1, n_mels, n_frames)
        mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]

        return mel_spec.astype(np.float32)

    def _infer_single_window(self, audio_window):
        """
        Run inference on a single audio window.

        Args:
            audio_window: Audio array (sample_length samples)

        Returns:
            Probability array for all classes
        """
        input_tensor = self.preprocess_audio(audio_window)
        logits = self.model(input_tensor, training=False)
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        return probabilities

    # =========================================================================
    # PROFESSOR FIX 3: Aggregation with max/percentile instead of mean
    # =========================================================================
    def _aggregate_predictions(self, all_probs, method='percentile'):
        """
        Aggregate predictions from multiple windows.

        PROFESSOR FIX 3: Mean dilutes cry signal with silence windows.
        Use max or percentile to capture "was there a cry at ANY point?"

        Args:
            all_probs: List of probability arrays
            method: 'max', 'percentile', or 'mean'

        Returns:
            Aggregated probability array
        """
        if not all_probs:
            return np.array([1.0 / self.num_classes] * self.num_classes)

        all_probs = np.array(all_probs)

        if method == 'max':
            # Take max probability for each class
            return np.max(all_probs, axis=0)
        elif method == 'percentile':
            # Take 90th percentile (robust to outliers, captures peak)
            return np.percentile(all_probs, AGGREGATION_PERCENTILE, axis=0)
        else:  # mean (not recommended but available)
            return np.mean(all_probs, axis=0)

    def predict(self, audio_path, use_sliding_window=True):
        """
        Run inference on an audio file.

        Args:
            audio_path: Path to audio file
            use_sliding_window: If True, use multi-window inference with aggregation

        Returns:
            dict with prediction results
        """
        # Load audio (with per-file RMS normalization)
        audio = self.load_audio(audio_path, normalize_per_file=True)

        start_time = time.perf_counter()

        if use_sliding_window and len(audio) > self.sample_length:
            # Multi-window inference with energy filtering
            windows = self._extract_sliding_windows(audio)

            all_probs = []
            window_details = []

            for i, (window, start_idx, rms) in enumerate(windows):
                probs = self._infer_single_window(window)
                all_probs.append(probs)
                window_details.append({
                    'start_ms': start_idx / self.sample_rate * 1000,
                    'rms': float(rms),
                    'probs': probs.tolist()
                })

                # DIAGNOSTIC: Log per-window class probabilities
                if self.verbose:
                    prob_str = " | ".join([f"{self.class_names[j]}: {probs[j]*100:5.1f}%"
                                           for j in range(len(self.class_names))])
                    winner = self.class_names[np.argmax(probs)]
                    print(f"    [Window {i+1:2d}] {prob_str} → {winner}")

            # PROFESSOR FIX 3: Aggregate using configurable mode
            probabilities = self._aggregate_predictions(all_probs, method=AGGREGATION_MODE)
            n_windows = len(windows)

            # DIAGNOSTIC: Summary of class distribution across windows
            if self.verbose and all_probs:
                all_probs_arr = np.array(all_probs)
                print(f"\n  [Class Distribution Summary]")
                for j, cls_name in enumerate(self.class_names):
                    cls_probs = all_probs_arr[:, j]
                    wins = np.sum(np.argmax(all_probs_arr, axis=1) == j)
                    print(f"    {cls_name:12s}: mean={cls_probs.mean()*100:5.1f}%, "
                          f"max={cls_probs.max()*100:5.1f}%, "
                          f"wins={wins}/{n_windows} windows")
                print(f"  [Aggregated ({AGGREGATION_MODE})]: {self.class_names[np.argmax(probabilities)]} "
                      f"({probabilities.max()*100:.1f}%)\n")

        else:
            # Single window inference (use max-energy window)
            audio_window = self._select_max_energy_window(audio, self.sample_length)
            probabilities = self._infer_single_window(audio_window)
            n_windows = 1
            window_details = []

        inference_time = (time.perf_counter() - start_time) * 1000

        # Get prediction
        class_id = np.argmax(probabilities)
        confidence = probabilities[class_id]
        class_name = self.class_names[class_id]

        # PROFESSOR FIX 4: Confidence calibration info
        prob_stats = {
            'mean': float(np.mean([w['probs'][0] for w in window_details])) if window_details else float(probabilities[0]),
            'max': float(np.max([w['probs'][0] for w in window_details])) if window_details else float(probabilities[0]),
            'std': float(np.std([w['probs'][0] for w in window_details])) if window_details else 0.0,
        }

        return {
            'class_id': int(class_id),
            'class_name': class_name,
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'inference_time_ms': float(inference_time),
            'audio_duration_ms': len(audio) / self.sample_rate * 1000,
            'n_windows': n_windows,
            'baby_cry_stats': prob_stats,  # PROFESSOR FIX 4
            'is_baby_cry': class_name == 'baby cry' and confidence >= CONFIDENCE_THRESHOLD,
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
                result['file'] = str(audio_path.name)
                results.append(result)

                if self.verbose:
                    cry_marker = " <-- BABY CRY!" if result['is_baby_cry'] else ""
                    print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}: "
                          f"{result['class_name']} ({result['confidence']*100:.1f}%){cry_marker}")
            except Exception as e:
                print(f"  [{i+1}/{len(audio_files)}] ERROR processing {audio_path.name}: {e}")

        return results

    def predict_realtime(self, duration_sec=None, callback=None):
        """
        Real-time inference from microphone with energy filtering.

        Args:
            duration_sec: Duration to record (None for continuous)
            callback: Function to call with each prediction result
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("ERROR: sounddevice not installed. Install with: pip install sounddevice")
            return

        print(f"\n{'='*60}")
        print("REAL-TIME BABY CRY DETECTION")
        print(f"{'='*60}")
        print(f"  Listening for {self.sample_length_ms}ms windows...")
        print(f"  Energy threshold: {ENERGY_THRESHOLD} (skipping silence)")
        print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        buffer = np.zeros(self.sample_length, dtype=np.float32)

        # Track recent predictions for smoothing
        recent_probs = []
        max_recent = 5

        def audio_callback(indata, frames, time_info, status):
            nonlocal buffer

            if status:
                print(f"Audio status: {status}")

            # Shift buffer and add new data
            new_samples = indata[:, 0] if len(indata.shape) > 1 else indata
            buffer = np.roll(buffer, -len(new_samples))
            buffer[-len(new_samples):] = new_samples

        try:
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            ):
                start_time = time.time()

                while True:
                    # Check duration
                    if duration_sec and (time.time() - start_time) > duration_sec:
                        break

                    # Run inference every 500ms
                    time.sleep(0.5)

                    # PROFESSOR FIX 1: Check energy before inference
                    has_energy, rms = self._check_window_energy(buffer)

                    if not has_energy:
                        print(f"\r  [{'.'*30}] (silence - rms={rms:.4f})         ", end='')
                        continue

                    # PROFESSOR FIX 2: RMS normalize the buffer
                    normalized_buffer = self._normalize_audio_rms(buffer.copy())

                    # Run inference
                    probabilities = self._infer_single_window(normalized_buffer)

                    # Track recent predictions
                    recent_probs.append(probabilities)
                    if len(recent_probs) > max_recent:
                        recent_probs.pop(0)

                    # PROFESSOR FIX 3: Use max of recent windows
                    smoothed_probs = np.max(recent_probs, axis=0)

                    class_id = np.argmax(smoothed_probs)
                    confidence = smoothed_probs[class_id]
                    class_name = self.class_names[class_id]

                    # Display result
                    bar_length = int(confidence * 30)
                    bar = '=' * bar_length + ' ' * (30 - bar_length)

                    # PROFESSOR FIX 4: Use calibrated threshold
                    if class_name == 'baby cry' and confidence >= CONFIDENCE_THRESHOLD:
                        print(f"\r  [{bar}] *** BABY CRY ***: {confidence*100:5.1f}%  ", end='')
                    else:
                        print(f"\r  [{bar}] {class_name}: {confidence*100:5.1f}%  ", end='')

                    # Callback
                    if callback:
                        callback({
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'probabilities': smoothed_probs,
                            'is_baby_cry': class_name == 'baby cry' and confidence >= CONFIDENCE_THRESHOLD,
                            'rms': float(rms)
                        })

        except KeyboardInterrupt:
            print("\n\nStopped.")


def print_result(result):
    """Pretty print a prediction result."""
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULT")
    print(f"{'='*60}")

    if 'file' in result:
        print(f"  File: {result['file']}")

    print(f"  Duration: {result['audio_duration_ms']:.0f} ms")
    print(f"  Windows processed: {result['n_windows']}")
    print(f"  Inference time: {result['inference_time_ms']:.2f} ms")

    print(f"\n  Predicted: {result['class_name'].upper()}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")

    # PROFESSOR FIX 4: Show calibration info
    if result['is_baby_cry']:
        print(f"\n  *** BABY CRY DETECTED ***")

    print(f"\n  Baby Cry Stats (across windows):")
    print(f"    Mean prob: {result['baby_cry_stats']['mean']*100:.1f}%")
    print(f"    Max prob:  {result['baby_cry_stats']['max']*100:.1f}%")
    print(f"    Std dev:   {result['baby_cry_stats']['std']*100:.1f}%")

    print(f"\n  All probabilities:")
    for class_name, prob in result['probabilities'].items():
        bar_length = int(prob * 30)
        bar = '=' * bar_length + ' ' * (30 - bar_length)
        marker = " <--" if class_name == result['class_name'] else ""
        print(f"    {class_name:12s} [{bar}] {prob*100:5.1f}%{marker}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SILICON-10 Baby Cry Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --audio test.wav
  python inference.py --dir audio_folder/
  python inference.py --realtime
  python inference.py --audio test.wav --model models/best_model.keras

Professor Fixes Applied:
  1. Energy threshold for window filtering (domain alignment)
  2. RMS normalization instead of peak normalization
  3. Max/percentile aggregation instead of mean
  4. Confidence calibration and logging
        """
    )

    parser.add_argument('--audio', type=str, help='Path to audio file for inference')
    parser.add_argument('--dir', type=str, help='Directory of audio files for batch inference')
    parser.add_argument('--realtime', action='store_true', help='Real-time microphone inference')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model file')
    parser.add_argument('--duration', type=float, default=None, help='Duration for real-time (seconds)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.audio, args.dir, args.realtime]):
        parser.print_help()
        print("\nError: Please specify --audio, --dir, or --realtime")
        sys.exit(1)

    # Initialize inference engine
    engine = BabyCryInference(model_path=args.model, verbose=not args.quiet)

    # Run inference
    if args.audio:
        result = engine.predict(args.audio)
        print_result(result)

    elif args.dir:
        results = engine.predict_batch(args.dir)

        # Summary statistics
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

            # Baby cry detections (with calibrated threshold)
            baby_cry_count = sum(1 for r in results if r['is_baby_cry'])
            print(f"\n  Baby Cry Detections (conf >= {CONFIDENCE_THRESHOLD*100:.0f}%): {baby_cry_count}")

            # Average confidence
            avg_conf = np.mean([r['confidence'] for r in results])
            print(f"\n  Average confidence: {avg_conf*100:.1f}%")

            # Average inference time
            avg_time = np.mean([r['inference_time_ms'] for r in results])
            print(f"  Average inference time: {avg_time:.2f} ms")
            print(f"{'='*60}\n")

    elif args.realtime:
        engine.predict_realtime(duration_sec=args.duration)


if __name__ == '__main__':
    main()
