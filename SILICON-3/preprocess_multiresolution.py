"""
Multi-Resolution Audio Preprocessing with Gammatone + Mel + Delta Features

Combines:
1. Mel-spectrogram (16 bins): Perceptual frequency representation
2. Gammatone filterbank (16 bins): Biologically-inspired transient detection
3. Delta features: Temporal dynamics (velocity of features)

Total: (16 Mel + 16 Gammatone) × 2 (static + delta) = 64 feature channels

For edge deployment on Silicon Labs xG26:
- Input: 1-second audio windows (16 kHz)
- Output: 64 × 100 feature matrix (25.6 KB per sample)
- Memory efficient for 512 KB RAM constraint
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class GammatoneFilterbank:
    """
    Gammatone filterbank - models human cochlea response.
    Better for transient detection than mel scale.

    Reference: Patterson et al. (1988), "Complex Sounds and Auditory Images"
    """

    def __init__(self, n_filters=16, sr=16000, fmin=50, fmax=8000):
        self.n_filters = n_filters
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax

        # Generate center frequencies (ERB scale - perceptually uniform)
        self.center_freqs = self._erb_space(fmin, fmax, n_filters)

    def _erb_space(self, low_freq, high_freq, n):
        """Generate n frequencies equally spaced on ERB scale"""
        # ERB (Equivalent Rectangular Bandwidth) formula
        ear_q = 9.26449  # Glasberg and Moore parameters
        min_bw = 24.7

        # Convert to ERB scale
        low_erb = ear_q * np.log(1 + low_freq / (ear_q * min_bw))
        high_erb = ear_q * np.log(1 + high_freq / (ear_q * min_bw))

        # Linear spacing on ERB scale
        erb_points = np.linspace(low_erb, high_erb, n)

        # Convert back to Hz
        freqs = (np.exp(erb_points / ear_q) - 1) * ear_q * min_bw
        return freqs

    def __call__(self, audio):
        """
        Apply gammatone filterbank to audio

        Args:
            audio: (n_samples,) audio waveform

        Returns:
            gammatone_spec: (n_filters, n_frames) gammatone spectrogram
        """
        n_fft = 512
        hop_length = 160  # 10ms hop

        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                           window='hann', center=True)
        magnitude = np.abs(stft)

        # Create gammatone-like filterbank weights
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        filterbank = np.zeros((self.n_filters, len(freqs)))

        for i, cf in enumerate(self.center_freqs):
            # Gammatone filter approximation using Gaussian-like shape
            # More asymmetric than mel filters (better for onsets)
            erb = 24.7 + 0.108 * cf  # ERB bandwidth

            # Asymmetric filter shape (models cochlear response)
            lower_slope = 4.0  # Steeper on low side
            upper_slope = 2.0  # Gentler on high side

            # Create asymmetric filter
            mask_low = freqs <= cf
            mask_high = freqs > cf

            filterbank[i, mask_low] = np.exp(
                -lower_slope * ((freqs[mask_low] - cf) / erb) ** 2
            )
            filterbank[i, mask_high] = np.exp(
                -upper_slope * ((freqs[mask_high] - cf) / erb) ** 2
            )

        # Apply filterbank
        gammatone_spec = np.dot(filterbank, magnitude)

        return gammatone_spec


def compute_delta(features, width=9):
    """
    Compute delta (velocity) features using regression formula

    Args:
        features: (n_features, n_frames) feature matrix
        width: window width for delta computation (typically 9)

    Returns:
        delta: (n_features, n_frames) delta features
    """
    n_features, n_frames = features.shape
    delta = np.zeros_like(features)

    # Regression coefficients
    denominator = 2 * sum([i**2 for i in range(1, width + 1)])

    for t in range(n_frames):
        numerator = np.zeros(n_features)

        for i in range(1, width + 1):
            # Forward difference
            if t + i < n_frames:
                numerator += i * features[:, t + i]
            else:
                numerator += i * features[:, -1]  # Pad with last frame

            # Backward difference
            if t - i >= 0:
                numerator -= i * features[:, t - i]
            else:
                numerator -= i * features[:, 0]  # Pad with first frame

        delta[:, t] = numerator / denominator

    return delta


def extract_multiresolution_features(audio_path, sr=16000, n_mels=16, n_gamma=16,
                                     hop_length=160, n_fft=512, fmin=50, fmax=8000):
    """
    Extract multi-resolution features: Mel + Gammatone + Delta

    Args:
        audio_path: Path to audio file
        sr: Sample rate (16 kHz for edge deployment)
        n_mels: Number of mel bins (16)
        n_gamma: Number of gammatone filters (16)
        hop_length: Hop length in samples (160 = 10ms)
        n_fft: FFT size
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        features: (64, n_frames) combined feature matrix
                  [0:16] = Mel static
                  [16:32] = Gammatone static
                  [32:48] = Mel delta
                  [48:64] = Gammatone delta
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    # 1. MEL-SPECTROGRAM (16 bins)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Energy
    )

    # Log compression (standard for mel)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # 2. GAMMATONE FILTERBANK (16 bins)
    gammatone_bank = GammatoneFilterbank(
        n_filters=n_gamma,
        sr=sr,
        fmin=fmin,
        fmax=fmax
    )
    gamma_spec = gammatone_bank(audio)

    # Log compression (same as mel for consistency)
    gamma_spec_db = librosa.power_to_db(gamma_spec, ref=np.max, top_db=80.0)

    # Ensure same length (padding if needed)
    min_len = min(mel_spec_db.shape[1], gamma_spec_db.shape[1])
    mel_spec_db = mel_spec_db[:, :min_len]
    gamma_spec_db = gamma_spec_db[:, :min_len]

    # 3. COMPUTE DELTA FEATURES
    mel_delta = compute_delta(mel_spec_db, width=9)
    gamma_delta = compute_delta(gamma_spec_db, width=9)

    # 4. STACK ALL FEATURES
    # Shape: (64, n_frames) = (16+16+16+16, n_frames)
    features = np.vstack([
        mel_spec_db,      # [0:16]   Mel static
        gamma_spec_db,    # [16:32]  Gammatone static
        mel_delta,        # [32:48]  Mel delta
        gamma_delta       # [48:64]  Gammatone delta
    ])

    return features.astype(np.float32)


def preprocess_dataset(dataset_dir, output_dir, max_time_frames=100):
    """
    Preprocess entire dataset with multi-resolution features

    Args:
        dataset_dir: Root directory with Baby_cry/ and dog/ subdirectories
        output_dir: Output directory for preprocessed .npy files
        max_time_frames: Maximum number of time frames (100 = 1 second)
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = ['Baby_cry', 'dog']

    for class_name in classes:
        class_input_dir = os.path.join(dataset_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Get all wav files
        wav_files = list(Path(class_input_dir).glob('*.wav'))

        print(f"\nProcessing {class_name} ({len(wav_files)} files)...")

        for wav_file in tqdm(wav_files, desc=f"  {class_name}"):
            try:
                # Extract features
                features = extract_multiresolution_features(str(wav_file))

                # Pad or truncate to max_time_frames
                n_frames = features.shape[1]

                if n_frames < max_time_frames:
                    # Pad with zeros
                    pad_width = max_time_frames - n_frames
                    features = np.pad(features, ((0, 0), (0, pad_width)),
                                     mode='constant', constant_values=0)
                else:
                    # Truncate to max_time_frames
                    features = features[:, :max_time_frames]

                # Save as .npy
                output_file = os.path.join(class_output_dir, wav_file.stem + '.npy')
                np.save(output_file, features)

            except Exception as e:
                print(f"  Error processing {wav_file.name}: {e}")
                continue

        print(f"  ✓ Completed {class_name}")

    print(f"\n✓ All preprocessing complete! Saved to: {output_dir}")


def visualize_multiresolution_features(output_dir, n_samples=3):
    """
    Create visualizations comparing all feature types
    """
    viz_dir = os.path.join(output_dir, 'visualizations_multiresolution')
    os.makedirs(viz_dir, exist_ok=True)

    classes = ['Baby_cry', 'dog']

    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        files = list(Path(class_dir).glob('*.npy'))[:n_samples]

        for i, file in enumerate(files):
            features = np.load(file)  # Shape: (64, 100)

            # Extract components
            mel_static = features[0:16, :]
            gamma_static = features[16:32, :]
            mel_delta = features[32:48, :]
            gamma_delta = features[48:64, :]

            # Create 4-panel visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Mel static
            im1 = axes[0, 0].imshow(mel_static, aspect='auto', origin='lower',
                                    cmap='viridis', interpolation='nearest')
            axes[0, 0].set_title(f'Mel-Spectrogram (16 bins)\n{class_name}', fontsize=12)
            axes[0, 0].set_ylabel('Mel Bins', fontsize=10)
            axes[0, 0].set_xlabel('Time Frames', fontsize=10)
            plt.colorbar(im1, ax=axes[0, 0])

            # Gammatone static
            im2 = axes[0, 1].imshow(gamma_static, aspect='auto', origin='lower',
                                    cmap='plasma', interpolation='nearest')
            axes[0, 1].set_title(f'Gammatone Filterbank (16 bins)\n{class_name}', fontsize=12)
            axes[0, 1].set_ylabel('Gammatone Channels', fontsize=10)
            axes[0, 1].set_xlabel('Time Frames', fontsize=10)
            plt.colorbar(im2, ax=axes[0, 1])

            # Mel delta
            im3 = axes[1, 0].imshow(mel_delta, aspect='auto', origin='lower',
                                    cmap='RdBu_r', interpolation='nearest',
                                    vmin=-20, vmax=20)
            axes[1, 0].set_title('Mel Delta (Velocity)', fontsize=12)
            axes[1, 0].set_ylabel('Mel Bins', fontsize=10)
            axes[1, 0].set_xlabel('Time Frames', fontsize=10)
            plt.colorbar(im3, ax=axes[1, 0])

            # Gammatone delta
            im4 = axes[1, 1].imshow(gamma_delta, aspect='auto', origin='lower',
                                    cmap='RdBu_r', interpolation='nearest',
                                    vmin=-20, vmax=20)
            axes[1, 1].set_title('Gammatone Delta (Velocity)', fontsize=12)
            axes[1, 1].set_ylabel('Gammatone Channels', fontsize=10)
            axes[1, 1].set_xlabel('Time Frames', fontsize=10)
            plt.colorbar(im4, ax=axes[1, 1])

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{class_name}_{i+1}_multiresolution.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\n✓ Visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    # Configuration
    DATASET_DIR = '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-3/dataset'
    OUTPUT_DIR = '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-3/processed_dataset_multiresolution_8s'

    # Preprocessing parameters
    SAMPLE_RATE = 16000
    N_MELS = 16           # Mel bins (reduced from 32 to make room for gammatone)
    N_GAMMATONE = 16      # Gammatone filters
    HOP_LENGTH = 160      # 10ms hop for streaming
    MAX_TIME_FRAMES = 800 # 8 second windows (INCREASED from 100)

    print("=" * 60)
    print("MULTI-RESOLUTION AUDIO PREPROCESSING")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nFeature Configuration:")
    print(f"  Mel bins: {N_MELS}")
    print(f"  Gammatone filters: {N_GAMMATONE}")
    print(f"  Delta features: YES")
    print(f"  Total channels: {N_MELS + N_GAMMATONE + N_MELS + N_GAMMATONE} (static + delta)")
    print(f"  Time frames: {MAX_TIME_FRAMES} (8.0 seconds)")
    print(f"  Hop length: {HOP_LENGTH} samples (10ms)")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print("=" * 60)

    # Run preprocessing
    preprocess_dataset(DATASET_DIR, OUTPUT_DIR, MAX_TIME_FRAMES)

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_multiresolution_features(OUTPUT_DIR, n_samples=3)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Check visualizations in: {OUTPUT_DIR}/visualizations_multiresolution/")
    print(f"2. Update config.py to use: {OUTPUT_DIR}")
    print(f"3. Train model with 64-channel input")
    print("=" * 60)
