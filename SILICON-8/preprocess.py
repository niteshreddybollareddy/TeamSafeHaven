"""
SILICON-8 Audio Preprocessing (Manual Implementation)
Based on Silicon Labs MLTK AudioFeatureGenerator approach

This is a standalone implementation that doesn't require silabs-mltk package.
It replicates the MLTK AudioFeatureGenerator logic for mel-spectrogram extraction.
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from pathlib import Path
import config


class AudioPreprocessor:
    """
    Manual implementation of audio preprocessing based on MLTK approach.

    This extracts mel-spectrograms using the same logic as MLTK AudioFeatureGenerator
    but without requiring the silabs-mltk package installation.
    """

    def __init__(self, max_duration_sec=10.0):
        """
        Initialize the audio preprocessor with config settings.

        Args:
            max_duration_sec: Maximum audio duration in seconds (default: 10.0)
                             Audio longer than this will be trimmed to avoid excessive memory usage.
        """
        self.sample_rate = config.SAMPLE_RATE_HZ
        self.sample_length_ms = config.SAMPLE_LENGTH_MS
        self.sample_length = config.SAMPLE_LENGTH

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

        print(f"✓ AudioPreprocessor initialized (Manual Mode)")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Max duration: {self.max_duration_sec}s ({self.max_samples} samples)")
        print(f"  Window size: {self.window_size_ms} ms ({self.window_size_length} samples)")
        print(f"  Window step: {self.window_step_ms} ms ({self.window_step_length} samples)")
        print(f"  FFT length: {self.n_fft}")
        print(f"  Mel bands: {self.n_mels}")
        print(f"  Frequency range: {self.fmin}-{self.fmax} Hz")

    def load_audio_file(self, audio_path, target_length=None, debug=False):
        """
        Load an audio file and preprocess it.

        Args:
            audio_path: Path to audio file
            target_length: Target length in samples (default: None, uses natural length up to max_samples)
            debug: If True, print debug information

        Returns:
            (audio_samples, sample_rate) tuple
        """
        try:
            # Load audio file
            audio, orig_sr = sf.read(audio_path, always_2d=False)
            if debug:
                print(f"  [1] Loaded: {len(audio)} samples @ {orig_sr}Hz ({len(audio)/orig_sr:.2f}s)")

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                if debug:
                    print(f"  [2] Stereo→Mono: {len(audio)} samples")

            # Resample if needed
            if orig_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
                if debug:
                    print(f"  [3] Resampled {orig_sr}→{self.sample_rate}Hz: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")

            # Apply DC notch filter if enabled
            if self.dc_filter:
                audio = self._apply_dc_notch_filter(audio)
                if debug:
                    print(f"  [4] DC filter applied")

            # Trim silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
            if debug:
                print(f"  [5] Silence trimmed: {len(audio)} → {len(audio_trimmed)} samples ({len(audio_trimmed)/self.sample_rate:.2f}s)")

            # IMPORTANT: Trim to max duration if longer (prevents memory issues)
            if len(audio_trimmed) > self.max_samples:
                # Take first max_samples (or could use random crop)
                audio = audio_trimmed[:self.max_samples]
                if debug:
                    print(f"  [6] ⚠ TRIMMED to max duration: {len(audio_trimmed)} → {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
            else:
                audio = audio_trimmed
                if debug:
                    print(f"  [6] No trimming needed (within {self.max_duration_sec}s limit)")

            # If specific target_length is provided, adjust to that length
            if target_length is not None:
                if len(audio) > target_length:
                    # Random crop
                    max_offset = len(audio) - target_length
                    offset = np.random.randint(0, max_offset + 1)
                    audio = audio[offset:offset + target_length]
                    if debug:
                        print(f"  [7] Random crop: offset={offset}, length={target_length}")
                elif len(audio) < target_length:
                    # Pad with zeros
                    pad_length = target_length - len(audio)
                    audio = np.pad(audio, (0, pad_length), mode='constant')
                    if debug:
                        print(f"  [7] Zero-padded: {len(audio)-pad_length} → {len(audio)} samples")

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                if debug:
                    print(f"  [8] Normalized: range [{audio.min():.3f}, {audio.max():.3f}]")

            return audio, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """
        Apply DC notch filter to remove DC offset.
        H(z) = (1 - z^-1) / (1 - coefficient*z^-1)

        Args:
            audio: Input audio signal
            coefficient: Filter coefficient (default: 0.95)

        Returns:
            Filtered audio
        """
        # Simple first-order IIR filter
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

    def extract_mel_spectrogram(self, audio, debug=False):
        """
        Extract mel-spectrogram from audio using librosa.

        This follows the MLTK approach:
        1. Compute STFT
        2. Convert to power spectrogram
        3. Apply mel filterbank
        4. Convert to log scale

        Args:
            audio: Audio signal (float32, shape: [sample_length])
            debug: If True, print debug information

        Returns:
            Mel-spectrogram (shape: [n_mels, n_time_frames]) - MLTK format (64, T)
        """
        if debug:
            print(f"  [9] Audio samples: {len(audio)} ({len(audio)/self.sample_rate:.2f}s)")

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
        if debug:
            print(f"  [10] STFT shape: {stft.shape} (freq_bins, time_frames)")

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
        if debug:
            print(f"  [11] Mel-spectrogram shape: {mel_spec.shape} (mel_bins, time_frames)")

        # Apply noise reduction if enabled
        if self.noise_reduction:
            mel_spec = self._apply_noise_reduction(mel_spec)
            if debug:
                print(f"  [12] Noise reduction applied")

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        if debug:
            print(f"  [13] Log scale (dB): range [{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}]")

        # Keep in MLTK internal format: (mel_bins, time_frames)
        # Do NOT transpose - save as (64, T) to match dataset.py expectation of (1, 64, T)
        if debug:
            print(f"  [14] Final shape (MLTK format): {mel_spec_db.shape} (mel_bins=64, time_frames=T)")

        return mel_spec_db

    def _apply_noise_reduction(self, mel_spec):
        """
        Simple noise reduction by subtracting noise floor estimate.

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

    def preprocess_file(self, audio_path, return_raw=False, debug=False):
        """
        Complete preprocessing pipeline: load audio -> extract features.

        Args:
            audio_path: Path to audio file
            return_raw: If True, return raw mel-spectrogram (float32)
                       If False, normalize for model input
            debug: If True, print debug information

        Returns:
            Mel-spectrogram features
        """
        # Load audio
        audio, sr = self.load_audio_file(audio_path, debug=debug)

        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio, debug=debug)

        if return_raw:
            return mel_spec

        # Normalize to zero mean, unit variance (will be done globally later)
        return mel_spec

    def process_batch(self, audio_paths):
        """
        Process a batch of audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Batch of mel-spectrograms (shape: [batch_size, n_time_frames, n_mels])
        """
        batch = []
        for audio_path in audio_paths:
            try:
                mel_spec = self.preprocess_file(audio_path)
                batch.append(mel_spec)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Add zero-filled spectrogram as fallback
                zero_spec = np.zeros((config.N_TIME_FRAMES, self.n_mels), dtype=np.float32)
                batch.append(zero_spec)

        return np.array(batch, dtype=np.float32)


def compute_normalization_stats(dataset_dir):
    """
    Compute mean and std for normalization from entire dataset.

    This should be done once on the training set and saved.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        (mean, std) tuple
    """
    print("\nComputing normalization statistics from dataset...")

    preprocessor = AudioPreprocessor()
    dataset_path = Path(dataset_dir)

    # Collect all spectrograms
    all_specs = []

    for class_name in config.CLASS_NAMES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"⚠ Warning: Class directory not found: {class_dir}")
            continue

        audio_files = list(class_dir.glob('*.wav')) + \
                     list(class_dir.glob('*.ogg')) + \
                     list(class_dir.glob('*.mp3'))

        print(f"  Processing {class_name}: {len(audio_files)} files...")

        for audio_file in audio_files:
            try:
                mel_spec = preprocessor.preprocess_file(str(audio_file), return_raw=True)
                all_specs.append(mel_spec)
            except Exception as e:
                print(f"    Error: {audio_file.name}: {e}")
                continue

    # Compute global mean and std
    all_specs = np.array(all_specs)
    mean = np.mean(all_specs)
    std = np.std(all_specs)

    print(f"\n✓ Normalization stats computed:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")

    # Save stats
    import os
    stats_path = os.path.join(config.MODEL_DIR, 'stats.npy')
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    np.save(stats_path, {'mean': mean, 'std': std})
    print(f"  Saved to: {stats_path}")

    return mean, std


def normalize_spectrogram(mel_spec, mean, std):
    """
    Normalize mel-spectrogram using mean and std.

    Args:
        mel_spec: Mel-spectrogram
        mean: Mean value
        std: Standard deviation

    Returns:
        Normalized spectrogram
    """
    return (mel_spec - mean) / (std + 1e-8)


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "="*60)
    print("Testing SILICON-8 Audio Preprocessing (Manual Mode)")
    print("="*60)

    preprocessor = AudioPreprocessor()

    # Generate synthetic audio for testing
    print("\nGenerating synthetic audio...")
    duration = 1.0
    t = np.linspace(0, duration, int(config.SAMPLE_RATE_HZ * duration))

    # Create a composite signal
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)

    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Extract mel-spectrogram
    print("\nExtracting mel-spectrogram...")
    mel_spec = preprocessor.extract_mel_spectrogram(audio)

    print(f"  Mel-spectrogram shape: {mel_spec.shape}")
    print(f"  Expected shape: {config.SPECTROGRAM_SHAPE}")
    print(f"  Mel-spectrogram range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")

    # Verify shape
    expected_shape = config.SPECTROGRAM_SHAPE
    if mel_spec.shape == expected_shape:
        print(f"\n✓ Shape matches config: {expected_shape}")
    else:
        print(f"\n✗ Shape mismatch! Expected {expected_shape}, got {mel_spec.shape}")
        # Try to fix the shape issue
        print(f"  Note: Actual frames: {mel_spec.shape[0]}, Expected: {expected_shape[0]}")

    print("\n" + "="*60)
    print("Preprocessing test completed!")
    print("="*60)


def process_dataset_to_npy():
    """
    Process all audio files from Dataset/ and save to processed_dataset/
    This replicates the MLTK preprocessing workflow.
    """
    import os
    from pathlib import Path
    from tqdm import tqdm

    print("\n" + "="*60)
    print("SILICON-8 DATASET PREPROCESSING")
    print("Manual Mode (Replicating MLTK AudioFeatureGenerator)")
    print("="*60)

    # Initialize preprocessor with 10s max duration
    preprocessor = AudioPreprocessor(max_duration_sec=10.0)

    # Input and output directories
    input_dir = Path(config.DATASET_DIR)
    output_dir = Path(config.PROCESSED_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\n⚠ NOTE: Audio files longer than {preprocessor.max_duration_sec}s will be trimmed to first {preprocessor.max_duration_sec}s")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each class
    total_processed = 0
    total_errors = 0

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        print(f"\n{'='*60}")
        print(f"Class: {class_name} (label: {class_idx})")
        print(f"{'='*60}")

        # Input and output class directories
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name

        if not class_input_dir.exists():
            print(f"⚠ Warning: Directory not found: {class_input_dir}")
            continue

        os.makedirs(class_output_dir, exist_ok=True)

        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.ogg', '*.mp3', '*.flac']:
            audio_files.extend(list(class_input_dir.glob(ext)))

        print(f"Found {len(audio_files)} audio files")

        # Process each file
        processed = 0
        errors = 0

        # Debug first 2 files in each class to show detailed processing
        debug_first_n = 2

        for idx, audio_file in enumerate(tqdm(audio_files, desc=f"Processing {class_name}")):
            try:
                # Enable debug for first few files
                debug_mode = (idx < debug_first_n)

                if debug_mode:
                    print(f"\n{'─'*70}")
                    print(f"DEBUG: File {idx+1}/{len(audio_files)}: {audio_file.name}")
                    print(f"{'─'*70}")

                # Extract mel-spectrogram (raw, not normalized)
                mel_spec = preprocessor.preprocess_file(str(audio_file), return_raw=True, debug=debug_mode)

                # Save as .npy file with shape (mel_bins, time_frames) = (64, T) - MLTK format
                output_file = class_output_dir / f"{audio_file.stem}.npy"
                np.save(output_file, mel_spec)

                if debug_mode:
                    print(f"  [15] Saved to: {output_file.name}")
                    print(f"  [16] File shape on disk: {mel_spec.shape}")
                    print(f"{'─'*70}\n")

                processed += 1

            except Exception as e:
                print(f"\n✗ Error: {audio_file.name}: {e}")
                errors += 1

        print(f"✓ Success: {processed}/{len(audio_files)}")
        if errors > 0:
            print(f"✗ Errors: {errors}")

        total_processed += processed
        total_errors += errors

    # Summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    print("Next step: python train.py")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--process':
        # Process entire dataset
        process_dataset_to_npy()
    else:
        # Test preprocessing
        test_preprocessing()
