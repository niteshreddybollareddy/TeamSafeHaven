"""
SILICON-14: PRO Preprocessing Pipeline
Hardware-Matched (xG26) + Pattern Aware (Max Energy Selection)
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import config
import os
import random

class ProPreprocessor:
    def __init__(self):
        self.settings = config
        
    def normalize_audio_loudness(self, y, target_rms=0.12):
        """Normalize audio to target RMS energy to match training average"""
        current_rms = np.sqrt(np.mean(y**2))
        if current_rms > 0:
            y = y * (target_rms / current_rms)
        return np.clip(y, -1.0, 1.0)
        
    def _apply_dc_notch_filter(self, audio):
        """Removes DC offset to match MLTK AFG behavior"""
        b = [1, -1]
        a = [1, -config.DC_NOTCH_FILTER_COEFFICIENT]
        return scipy.signal.lfilter(b, a, audio)
        
    def apply_spec_augment(self, mel_spec, time_mask_max=20, freq_mask_max=10):
        """
        SpecAugment: Randomly masks frequency and time bands.
        Forces the model to learn shapes, not specific frequencies.
        """
        mel_spec = mel_spec.copy()
        n_mels, n_frames = mel_spec.shape
        
        # Frequency masking
        f = np.random.randint(0, freq_mask_max)
        f0 = np.random.randint(0, n_mels - f)
        mel_spec[f0:f0 + f, :] = -80.0
        
        # Time masking
        t = np.random.randint(0, time_mask_max)
        t0 = np.random.randint(0, n_frames - t)
        mel_spec[:, t0:t0 + t] = -80.0
        
        return mel_spec

    def apply_mltk_augmentations(self, audio):
        """
        MLTK-style Augmentations:
        1. Random Gain (0.8x to 1.2x)
        2. Random White Noise (simulates background hiss)
        3. Random Time Shift
        """
        # A. Random Gain (matches audiomentations.Gain)
        gain = np.random.uniform(0.8, 1.2)
        audio = audio * gain

        # B. Random Noise (matches audiomentations.AddGaussianNoise)
        # Adding subtle white noise to make the model "tough"
        noise_level = np.random.uniform(0, 0.002)
        noise = np.random.normal(0, noise_level, len(audio))
        audio = audio + noise

        # C. Random Time Shift (simulates the sound starting at slightly different times)
        # Roll the audio by up to 100ms
        shift = np.random.randint(-1600, 1600)
        audio = np.roll(audio, shift)
        
        return np.clip(audio, -1.0, 1.0)

    def get_processed_audio(self, audio_path, augment=False):
        """
        Returns the raw audio signal after all preprocessing steps 
        (DC Notch, Augmentation, Cropping, Warming) but BEFORE STFT.
        """
        # 1. Load Audio and Standardize
        try:
            if audio_path.endswith('.mp3'):
                y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE_HZ)
            else:
                y, sr = sf.read(audio_path, always_2d=False)
        except Exception:
            y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE_HZ)
        
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        if sr != config.SAMPLE_RATE_HZ:
            y = librosa.resample(y, orig_sr=sr, target_sr=config.SAMPLE_RATE_HZ)
        
        # 1.5 NEW: Normalize audio loudness (Match training energy baseline)
        y = self.normalize_audio_loudness(y, target_rms=0.12)
        
        # 2. Hardware Notch Filter
        y = self._apply_dc_notch_filter(y)

        # 3. Apply MLTK Augmentations
        if augment:
            y = self.apply_mltk_augmentations(y)

        # 4. Energy-Based Window Selection
        target_len = config.SAMPLE_LENGTH
        if len(y) > target_len:
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512).squeeze()
            max_rms_idx = np.argmax(rms)
            center_sample = max_rms_idx * 512
            start = max(0, min(center_sample - target_len // 2, len(y) - target_len))
            y = y[start:start + target_len]
        else:
            pad_len = target_len - len(y)
            noise_pad = np.random.normal(0, 0.0005, pad_len)
            y = np.concatenate([y, noise_pad])

        # 5. NOISE REDUCTION WARM-UP
        noise_warmup = np.random.normal(0, 0.0008, config.WARMUP_SAMPLES)
        y_final = np.concatenate([noise_warmup, y])
        
        return y_final

    def extract_features(self, audio_path, augment=False):
        """
        Calibrated Pipeline for xG26 judges: 
        Load -> Augment (if training) -> NR Warm-up -> Energy Window -> Mel -> Log
        """
        y_for_stft = self.get_processed_audio(audio_path, augment=augment)

        # 6. Mel Spectrogram Extraction
        stft = librosa.stft(
            y_for_stft,
            n_fft=512,
            hop_length=config.WINDOW_STEP_LENGTH,
            win_length=config.WINDOW_SIZE_LENGTH,
            window='hann',
            center=True,
            pad_mode='constant'
        )
        
        power_spec = np.abs(stft) ** 2
        mel_basis = librosa.filters.mel(
            sr=config.SAMPLE_RATE_HZ,
            n_fft=512,
            n_mels=64,
            fmin=config.FILTERBANK_LOWER_BAND_LIMIT,
            fmax=config.FILTERBANK_UPPER_BAND_LIMIT
        )
        mel_spec = np.dot(mel_basis, power_spec)
        
        # Log Scale with top_db limit same as MLTK
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        
        # 7. Apply SpecAugment (Spectral-domain augmentation)
        if augment:
            mel_spec = self.apply_spec_augment(mel_spec)
        
        # 8. Discard warm-up frames to return clean (64, 148) shape
        mel_spec = mel_spec[:, -config.N_TIME_FRAMES:]
        
        return mel_spec

if __name__ == '__main__':
    prep = ProPreprocessor()
    print("✓ ProPreprocessor Hardware-Calibrated & Energy-Aware Ready")
