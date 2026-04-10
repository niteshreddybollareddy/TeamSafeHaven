"""
SILICON-12 Audio Preprocessing
Based on SILICON-9's PROVEN preprocessing pipeline (99.81% accuracy)

CRITICAL DESIGN DECISIONS (from SILICON-9):
1. Energy-based windowing for ALL classes (max-energy window)
2. Energy window BEFORE silence trimming (preserves signal onset)
3. Less aggressive silence trim (top_db=40)
4. 1.5s fixed window (148 time frames with 10ms hop, center=False)
5. Global z-score normalization (computed once, reused)

WHY SILICON-10 FAILED:
- Used sliding windows with averaging (diluted signal)
- Energy filtering AFTER windowing (could reject cry onset)
- RMS normalization per-file (mismatched training)
- 2-second windows with 5ms hop (too many frames)

This implementation ensures EXACT match between training and inference preprocessing.
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class AudioPreprocessor:
    """
    Audio preprocessor using SILICON-9's proven pipeline.

    Key features:
    - Max-energy windowing (captures loudest part of audio)
    - Fixed 1.5s windows (24000 samples @ 16kHz)
    - 64 mel bins x 148 time frames output (center=False STFT)
    - Deterministic preprocessing (no randomness during inference)
    """

    def __init__(self):
        """Initialize with SILICON-12 config (based on SILICON-9)."""
        self.sample_rate = config.SAMPLE_RATE_HZ
        self.sample_length_ms = config.SAMPLE_LENGTH_MS
        self.sample_length = config.SAMPLE_LENGTH  # 24000 samples for 1.5s

        self.window_size_ms = config.WINDOW_SIZE_MS
        self.window_step_ms = config.WINDOW_STEP_MS
        self.n_fft = config.FFT_LENGTH

        self.n_mels = config.FILTERBANK_N_CHANNELS
        self.fmin = config.FILTERBANK_LOWER_BAND_LIMIT
        self.fmax = config.FILTERBANK_UPPER_BAND_LIMIT

        self.noise_reduction = config.NOISE_REDUCTION_ENABLE
        self.dc_filter = config.DC_NOTCH_FILTER_ENABLE

        # Frame parameters
        self.window_size_length = config.WINDOW_SIZE_LENGTH  # 400
        self.window_step_length = config.WINDOW_STEP_LENGTH  # 160

        print(f"AudioPreprocessor initialized (SILICON-9 pipeline)")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Window: {self.sample_length_ms}ms ({self.sample_length} samples)")
        print(f"  STFT: {self.window_size_ms}ms size, {self.window_step_ms}ms step")
        print(f"  Output: ({self.n_mels}, {config.N_TIME_FRAMES})")

    def load_audio_file(self, audio_path, debug=False):
        """
        Load and preprocess audio file using ENERGY-BASED WINDOWING.

        CRITICAL: Always use max-energy window (SILICON-9's key insight)
        This ensures the model sees the loudest/most important part of audio.

        Args:
            audio_path: Path to audio file
            debug: Print debug information

        Returns:
            (audio_samples, sample_rate) tuple
        """
        try:
            # Load audio
            audio, orig_sr = sf.read(audio_path, always_2d=False)
            if debug:
                print(f"  [1] Loaded: {len(audio)} samples @ {orig_sr}Hz ({len(audio)/orig_sr:.2f}s)")

            # Stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                if debug:
                    print(f"  [2] Stereo->Mono: {len(audio)} samples")

            # Resample if needed
            if orig_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
                if debug:
                    print(f"  [3] Resampled {orig_sr}->{self.sample_rate}Hz: {len(audio)} samples")

            # DC notch filter
            if self.dc_filter:
                audio = self._apply_dc_notch_filter(audio)
                if debug:
                    print(f"  [4] DC filter applied")

            # SILICON-9 KEY: Energy window BEFORE silence trimming
            if len(audio) < self.sample_length:
                # Short audio: pad to target length
                pad_length = self.sample_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
                if debug:
                    print(f"  [5] Padded to {self.sample_length} samples")
            else:
                # Long audio: select max-energy window
                audio = self._select_max_energy_window(audio, self.sample_length, debug)
                if debug:
                    print(f"  [5] Max-energy window selected")

            # Light silence trim AFTER energy windowing (less aggressive)
            trim_db = config.SILENCE_TRIM_TOP_DB  # 40 dB
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=trim_db)

            # Re-pad if trimming made it too short
            if len(audio_trimmed) < self.sample_length:
                pad_length = self.sample_length - len(audio_trimmed)
                audio = np.pad(audio_trimmed, (0, pad_length), mode='constant')
                if debug:
                    print(f"  [6] Re-padded after trim: {len(audio)} samples")
            else:
                audio = audio_trimmed[:self.sample_length]
                if debug:
                    print(f"  [6] Trimmed (top_db={trim_db}): {len(audio)} samples")

            # Peak normalization
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                if debug:
                    print(f"  [7] Normalized: [{audio.min():.3f}, {audio.max():.3f}]")

            return audio, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Error loading {audio_path}: {e}")

    def _select_max_energy_window(self, audio, target_length, debug=False):
        """
        Select window with maximum energy (SILICON-9's key technique).

        This ensures baby cry / dog bark onset is captured, not silence.
        Applied to raw audio BEFORE any trimming.

        Args:
            audio: Raw audio signal
            target_length: Window length in samples
            debug: Print debug info

        Returns:
            Audio window with maximum energy
        """
        if len(audio) < target_length:
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

        # Safe clamping (SILICON-9 fix)
        start_idx = center_sample - target_length // 2
        start_idx = max(0, min(start_idx, len(audio) - target_length))
        end_idx = start_idx + target_length

        if debug:
            print(f"      Max energy at frame {max_energy_frame} (sample {center_sample})")
            print(f"      Window: [{start_idx}:{end_idx}]")

        return audio[start_idx:end_idx]

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """Apply DC notch filter to remove DC offset."""
        b = [1, -1]
        a = [1, -coefficient]
        return scipy.signal.lfilter(b, a, audio)

    def _apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter (boosts high frequencies)."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def extract_mel_spectrogram(self, audio, debug=False):
        """
        Extract mel-spectrogram matching SILICON-9's pipeline.

        Output: (64, 148) - 64 mel bins, 148 time frames for 1.5s audio

        Formula (center=False): n_frames = (n_samples - win_length) // hop_length + 1
                                         = (24000 - 400) // 160 + 1 = 148 frames

        Args:
            audio: Audio signal (float32, 24000 samples)
            debug: Print debug info

        Returns:
            Mel-spectrogram (shape: [n_mels, n_time_frames])
        """
        if debug:
            print(f"  [8] Audio: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")

        # Pre-emphasis
        audio = self._apply_preemphasis(audio)

        hop_length = self.window_step_length

        if debug:
            expected_frames = (len(audio) - self.window_size_length) // hop_length + 1
            print(f"  [8a] hop={hop_length}, expected frames={expected_frames}")

        # STFT with center=False for predictable frame count
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=self.window_size_length,
            window='hann',
            center=False  # CRITICAL: center=False for exact frame count
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

        # Noise reduction
        if self.noise_reduction:
            mel_spec = self._apply_noise_reduction(mel_spec)

        # Log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        if debug:
            print(f"  [11] Log dB: [{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}]")
            print(f"  [12] Final: {mel_spec_db.shape}")

        return mel_spec_db

    def _apply_noise_reduction(self, mel_spec):
        """Simple noise reduction by subtracting noise floor estimate."""
        noise_floor = np.median(mel_spec[:, :5], axis=1, keepdims=True)
        mel_spec_clean = mel_spec - noise_floor
        mel_spec_clean = np.maximum(
            mel_spec_clean,
            mel_spec * config.NOISE_REDUCTION_MIN_SIGNAL_REMAINING
        )
        return mel_spec_clean

    def preprocess_file(self, audio_path, return_raw=False, debug=False):
        """
        Complete preprocessing pipeline: load -> extract features.

        Args:
            audio_path: Path to audio file
            return_raw: If True, return raw mel-spectrogram
            debug: Print debug info

        Returns:
            Mel-spectrogram (64, 98) or None if error
        """
        audio, _ = self.load_audio_file(audio_path, debug=debug)
        mel_spec = self.extract_mel_spectrogram(audio, debug=debug)
        return mel_spec


def process_dataset_to_npy():
    """
    Process all audio files from Dataset/ and save to processed_dataset/
    Using SILICON-9's energy-based windowing strategy.
    """
    from tqdm import tqdm

    print("\n" + "="*60)
    print("SILICON-12 DATASET PREPROCESSING")
    print("Energy-Based Windowing (SILICON-9 Strategy)")
    print("="*60)

    preprocessor = AudioPreprocessor()

    input_dir = Path(config.DATASET_DIR)
    output_dir = Path(config.PROCESSED_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {input_dir}")

    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nStrategy: Max-energy window ({config.SAMPLE_LENGTH_MS}ms)")

    os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    total_errors = 0
    class_counts = {}

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        print(f"\n{'='*60}")
        print(f"Class: {class_name} (label: {class_idx})")
        print(f"{'='*60}")

        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name

        if not class_input_dir.exists():
            print(f"Warning: {class_input_dir} not found")
            continue

        os.makedirs(class_output_dir, exist_ok=True)

        # Find audio files (recursive search)
        audio_files = []
        for ext in ['*.wav', '*.ogg', '*.mp3', '*.flac']:
            audio_files.extend(list(class_input_dir.rglob(ext)))

        print(f"Found {len(audio_files)} files")

        processed = 0
        errors = 0
        debug_first_n = 2

        for idx, audio_file in enumerate(tqdm(audio_files, desc=f"Processing {class_name}")):
            try:
                debug_mode = (idx < debug_first_n)

                if debug_mode:
                    print(f"\n{'─'*60}")
                    print(f"DEBUG: {audio_file.name}")
                    print(f"{'─'*60}")

                mel_spec = preprocessor.preprocess_file(
                    str(audio_file),
                    return_raw=True,
                    debug=debug_mode
                )

                # Save as .npy (64, 98)
                output_file = class_output_dir / f"{audio_file.stem}.npy"
                np.save(output_file, mel_spec)

                if debug_mode:
                    print(f"  Saved: {output_file.name} {mel_spec.shape}")

                processed += 1

            except Exception as e:
                print(f"\nError: {audio_file.name}: {e}")
                errors += 1

        print(f"Success: {processed}/{len(audio_files)}")
        if errors > 0:
            print(f"Errors: {errors}")

        class_counts[class_name] = processed
        total_processed += processed
        total_errors += errors

    # Summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\nClass distribution:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    print(f"\nTotal: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"Output: {output_dir}")
    print(f"\nNext: python train.py")


def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("\n" + "="*60)
    print("SILICON-12 Preprocessing Test")
    print("="*60)

    preprocessor = AudioPreprocessor()

    # Generate test audio (1.5s sine wave)
    duration = 1.5
    t = np.linspace(0, duration, int(config.SAMPLE_RATE_HZ * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)

    print(f"\nTest audio: {len(audio)} samples ({duration}s)")

    mel_spec = preprocessor.extract_mel_spectrogram(audio, debug=True)

    print(f"\nOutput shape: {mel_spec.shape}")
    print(f"Expected: {config.SPECTROGRAM_SHAPE}")

    if mel_spec.shape == config.SPECTROGRAM_SHAPE:
        print(f"\nSHAPE MATCH")
    else:
        print(f"\nShape mismatch!")
        print(f"  Got {mel_spec.shape}, expected {config.SPECTROGRAM_SHAPE}")

    print("\n" + "="*60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--process':
        process_dataset_to_npy()
    else:
        test_preprocessing()
