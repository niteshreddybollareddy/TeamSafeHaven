"""
SILICON-13 Audio Preprocessing
3-Class Classification: Baby Cry, Dog, Unknown
Based on SILICON-9 preprocessing with energy-based windowing

CRITICAL FEATURES:
✓ Energy-based windowing for all classes (captures loudest part)
✓ Less aggressive silence trimming
✓ Global mean/std normalization
✓ 1.5s audio window (24000 samples @ 16kHz)
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
    Same as SILICON-9 but for 3 classes.
    """

    def __init__(self):
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

        self.window_size_length = int((self.window_size_ms * self.sample_rate) / 1000)
        self.window_step_length = int((self.window_step_ms * self.sample_rate) / 1000)

        print(f"✓ AudioPreprocessor initialized")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Target duration: {self.sample_length_ms}ms ({self.sample_length} samples)")
        print(f"  Window size: {self.window_size_ms} ms ({self.window_size_length} samples)")
        print(f"  Window step: {self.window_step_ms} ms ({self.window_step_length} samples)")
        print(f"  FFT length: {self.n_fft}")
        print(f"  Mel bands: {self.n_mels}")

    def load_audio_file(self, audio_path, target_length=None, debug=False):
        """
        Load audio file with energy-based windowing.
        """
        try:
            if target_length is None:
                target_length = self.sample_length

            # Load audio file
            audio, orig_sr = sf.read(audio_path, always_2d=False)
            if debug:
                print(f"  [1] Loaded: {len(audio)} samples @ {orig_sr}Hz")

            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if orig_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

            # Apply DC notch filter
            if self.dc_filter:
                audio = self._apply_dc_notch_filter(audio)

            # Energy-based windowing
            if len(audio) < target_length:
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
            else:
                audio = self._select_max_energy_window(audio, target_length, debug)

            # Light silence trimming
            trim_db = 40
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=trim_db)

            if len(audio_trimmed) < target_length:
                pad_length = target_length - len(audio_trimmed)
                audio = np.pad(audio_trimmed, (0, pad_length), mode='constant')
            else:
                audio = audio_trimmed[:target_length]

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    def _select_max_energy_window(self, audio, target_length, debug=False):
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

        if debug:
            print(f"      Max energy at frame {max_energy_frame}, window: {start_idx}-{end_idx}")

        return audio[start_idx:end_idx]

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """Apply DC notch filter."""
        b = [1, -1]
        a = [1, -coefficient]
        filtered = scipy.signal.lfilter(b, a, audio)
        return filtered

    def _apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def extract_mel_spectrogram(self, audio, debug=False):
        """Extract mel-spectrogram from audio."""
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

        # Apply noise reduction
        if self.noise_reduction:
            mel_spec = self._apply_noise_reduction(mel_spec)

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        return mel_spec_db

    def _apply_noise_reduction(self, mel_spec):
        """Simple noise reduction."""
        noise_floor = np.median(mel_spec[:, :5], axis=1, keepdims=True)
        mel_spec_clean = mel_spec - noise_floor
        mel_spec_clean = np.maximum(mel_spec_clean, mel_spec * config.NOISE_REDUCTION_MIN_SIGNAL_REMAINING)
        return mel_spec_clean

    def preprocess_file(self, audio_path, return_raw=False, debug=False):
        """Complete preprocessing pipeline."""
        audio, sr = self.load_audio_file(audio_path, debug=debug)
        mel_spec = self.extract_mel_spectrogram(audio, debug=debug)

        if return_raw:
            return mel_spec

        return mel_spec


def process_dataset_to_npy():
    """
    Process all audio files from raw directories and save to processed_dataset/
    For 3 classes: Baby Cry, Dog, Unknown
    """
    import os
    from tqdm import tqdm

    print("\n" + "="*60)
    print("SILICON-13 DATASET PREPROCESSING")
    print("3-Class: Baby Cry, Dog, Unknown")
    print("="*60)

    preprocessor = AudioPreprocessor()

    output_dir = Path(config.PROCESSED_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Map class names to their input directories
    # Adjust these paths based on your actual directory structure
    class_dirs = {
        'Baby Cry': os.path.join(config.BASE_DIR, 'baby cry'),
        'Dog': os.path.join(config.BASE_DIR, 'dog'),
        'Unknown': os.path.join(config.BASE_DIR, 'unknown')
    }

    print(f"\nInput directories:")
    for class_name, dir_path in class_dirs.items():
        print(f"  {class_name}: {dir_path}")
    print(f"Output: {output_dir}")

    total_processed = 0
    total_errors = 0
    class_counts = {}

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        print(f"\n{'='*60}")
        print(f"Class: {class_name} (label: {class_idx})")
        print(f"{'='*60}")

        class_input_dir = Path(class_dirs[class_name])
        class_output_dir = output_dir / class_name

        if not class_input_dir.exists():
            print(f"⚠ Warning: Directory not found: {class_input_dir}")
            continue

        os.makedirs(class_output_dir, exist_ok=True)

        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.ogg', '*.mp3', '*.flac']:
            audio_files.extend(list(class_input_dir.rglob(ext)))

        print(f"Found {len(audio_files)} audio files")

        processed = 0
        errors = 0
        debug_first_n = 2

        for idx, audio_file in enumerate(tqdm(audio_files, desc=f"Processing {class_name}")):
            try:
                debug_mode = (idx < debug_first_n)

                if debug_mode:
                    print(f"\n{'─'*70}")
                    print(f"DEBUG: File {idx+1}: {audio_file.name}")
                    print(f"{'─'*70}")

                mel_spec = preprocessor.preprocess_file(str(audio_file), return_raw=True, debug=debug_mode)

                output_file = class_output_dir / f"{audio_file.stem}.npy"
                np.save(output_file, mel_spec)

                if debug_mode:
                    print(f"  Saved to: {output_file.name}, shape: {mel_spec.shape}")

                processed += 1

            except Exception as e:
                print(f"\n✗ Error: {audio_file.name}: {e}")
                errors += 1

        print(f"✓ Success: {processed}/{len(audio_files)}")
        if errors > 0:
            print(f"✗ Errors: {errors}")

        class_counts[class_name] = processed
        total_processed += processed
        total_errors += errors

    # Summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\nClass counts:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count:,} samples")

    print(f"\nTotal processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    print("Next step: python train.py")


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "="*60)
    print("Testing SILICON-13 Audio Preprocessing")
    print("="*60)

    preprocessor = AudioPreprocessor()

    # Generate synthetic audio
    print("\nGenerating synthetic audio...")
    duration = 1.5
    t = np.linspace(0, duration, int(config.SAMPLE_RATE_HZ * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)

    print(f"  Audio shape: {audio.shape}")

    # Extract mel-spectrogram
    print("\nExtracting mel-spectrogram...")
    mel_spec = preprocessor.extract_mel_spectrogram(audio)

    print(f"  Mel-spectrogram shape: {mel_spec.shape}")
    print(f"  Expected shape: {config.SPECTROGRAM_SHAPE}")
    print(f"  Range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")

    print("\n" + "="*60)
    print("Preprocessing test completed!")
    print("="*60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--process':
        process_dataset_to_npy()
    else:
        test_preprocessing()
