import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import scipy.signal
import config

# ==========================================
# HARDWARE-READY DSP FUNCTIONS
# ==========================================

def hz_to_mel(frequencies):
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)

def mel_to_hz(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
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

def compute_stft_manual(y, n_fft, hop_length):
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

def power_to_db_manual(S, ref=1.0, amin=1e-10, top_db=80.0):
    S_log = 10.0 * np.log10(np.maximum(amin, S))
    S_log -= 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        S_log = np.maximum(S_log, S_log.max() - top_db)
    return S_log

# ==========================================
# HIGH-FIDELITY PREPROCESSING
# ==========================================

def apply_preemphasis(y, coeff=0.97):
    """
    Hardware-friendly High-pass filter.
    Boosts high frequencies to preserve details in cries/voices.
    Formula: y[t] = x[t] - coeff * x[t-1]
    """
    return np.append(y[0], y[1:] - coeff * y[:-1])

def load_and_preprocess_audio(audio_path, target_sr=config.SAMPLE_RATE):
    try:
        # 1. Load audio with SoundFile (Highest precision read)
        y, orig_sr = sf.read(audio_path, always_2d=False)

        # 2. Handle Multi-channel (Mix to Mono)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # 3. Clean Resampling (Polyphase)
        if orig_sr != target_sr:
            gcd = np.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            y = scipy.signal.resample_poly(y, up, down)

        # 4. Pre-emphasis
        y = apply_preemphasis(y)

        # 5. Less Aggressive Silence Trimming
        frame_len = 2048
        hop = 512
        y_sq = y**2

        # Compute energy
        energy = np.array([np.sum(y_sq[i:i+frame_len]) for i in range(0, len(y), hop)])

        if len(energy) > 0:
            max_energy = np.max(energy)
            if max_energy > 0:
                # -40dB threshold
                threshold = max_energy * (10**(-40/10))
                active_frames = np.where(energy > threshold)[0]

                if len(active_frames) > 0:
                    start = active_frames[0] * hop
                    end = min(len(y), (active_frames[-1] * hop) + frame_len)

                    # Add buffer around trim
                    start = max(0, start - hop*2)
                    end = min(len(y), end + hop*2)

                    y = y[start:end]

        # 6. Peak Normalization
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        return y, target_sr

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.zeros(0), target_sr

def extract_melspectrogram(audio_path, sr=config.SAMPLE_RATE, n_mels=config.N_MELS):
    y, sr = load_and_preprocess_audio(audio_path, sr)

    if len(y) == 0:
        return np.zeros((n_mels, config.MAX_TIME_FRAMES)), 0

    # Compute STFT
    stft_complex = compute_stft_manual(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    power_spec = np.abs(stft_complex)**2

    # Filterbank
    mel_basis = create_mel_filterbank(sr, config.N_FFT, n_mels, fmin=50, fmax=8000)
    mel_spec = np.dot(mel_basis, power_spec)

    # Log
    mel_spec_db = power_to_db_manual(mel_spec, ref=np.max(mel_spec))

    return mel_spec_db, len(y) / sr

def should_keep_file(duration, class_name):
    """
    Filter outliers based on class-specific thresholds
    """
    outlier_thresholds = {
        'Dog': 20.0,       # Remove > 20s
        'Unusual': None,   # Keep all (natural 7-8s)
        'Baby Cry': None,  # Keep all (natural ~7s with 1-2s cry bursts)
        'Unknown': 20.0    # Remove > 20s
    }

    threshold = outlier_thresholds.get(class_name)
    if threshold is None:
        return True
    return duration <= threshold

def visualize_sample(mel_spec, title, output_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    plt.title(title)
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time Frames')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_dataset():
    print("="*60)
    print("SILICON-7 Preprocessing (Natural Data - 16kHz Mono)")
    print("="*60)

    # Raw data directories
    raw_dirs = {
        'Dog': 'dog',
        'Unusual': 'unusual',
        'Baby Cry': 'baby cry',
        'Unknown': 'unknown'
    }

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    viz_dir = os.path.join(config.PROCESSED_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    all_stats = {'total': 0, 'errors': 0, 'outliers_removed': 0}

    for class_name, raw_dir in raw_dirs.items():
        input_dir = os.path.join(config.BASE_DIR, raw_dir)
        output_dir = os.path.join(config.PROCESSED_DIR, class_name)
        os.makedirs(output_dir, exist_ok=True)

        audio_files = list(Path(input_dir).glob('*.wav')) + list(Path(input_dir).glob('*.ogg'))
        print(f'\nProcessing {class_name}: {len(audio_files)} files')

        processed_count = 0
        outliers_count = 0

        for i, audio_file in enumerate(tqdm(audio_files, ncols=100)):
            try:
                # First, check duration to filter outliers
                info = sf.info(str(audio_file))
                duration = info.duration

                if not should_keep_file(duration, class_name):
                    outliers_count += 1
                    all_stats['outliers_removed'] += 1
                    continue

                # Extract mel-spectrogram
                mel_spec, duration = extract_melspectrogram(str(audio_file))

                # Save as-is for all classes (no segmentation)
                output_path = os.path.join(output_dir, audio_file.stem + '.npy')
                np.save(output_path, mel_spec)

                # Visualize first few
                if processed_count < 3:
                    viz_path = os.path.join(viz_dir, f'{class_name}_{processed_count}.png')
                    visualize_sample(mel_spec, f'{class_name} - {audio_file.name}', viz_path)

                processed_count += 1
                all_stats['total'] += 1

            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                all_stats['errors'] += 1

        print(f"Processed {processed_count} files for {class_name} (removed {outliers_count} outliers)")

    print("\n" + "="*60)
    print("Preprocessing Summary:")
    print(f"  Total files processed: {all_stats['total']}")
    print(f"  Outliers removed: {all_stats['outliers_removed']}")
    print(f"  Errors: {all_stats['errors']}")
    print("="*60)

if __name__ == '__main__':
    process_dataset()
