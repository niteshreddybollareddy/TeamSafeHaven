"""
SILICON-15: MLTK-Style Preprocessing
Implements the exact Audio Feature Generator (AFG) settings:
16kHz, 64-channel, 1.5s window, Log-Scale.
"""

import numpy as np
import librosa
import soundfile as sf

class AFGPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.sample_len_ms = 1500
        self.window_size_ms = 30
        self.window_step_ms = 10
        self.n_mels = 64
        
        # Internal pixel-math
        self.samples_per_window = int(self.sample_rate * (self.sample_len_ms / 1000))
        self.n_fft = 512
        self.hop_length = int(self.sample_rate * (self.window_step_ms / 1000))
        self.win_length = int(self.sample_rate * (self.window_size_ms / 1000))

    def _apply_dc_notch_filter(self, audio):
        """Silabs Hardware simulation: Removes DC offset (Coef 0.95)"""
        import scipy.signal
        b, a = [1, -1], [1, -0.95]
        return scipy.signal.lfilter(b, a, audio)

    def apply_augmentations(self, y):
        """Replicates MLTK ParallelAudioDataGenerator augmentations"""
        # 1. Random Gain (0.5 to 1.2)
        gain = np.random.uniform(0.5, 1.2)
        y = y * gain
        
        # 2. Add White Noise
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
        
        # 3. Random Time Shift (Small roll)
        shift = np.random.randint(int(self.sample_rate * 0.1))
        y = np.roll(y, shift)
        
        return np.clip(y, -1.0, 1.0)

    def _get_max_energy_window(self, y):
        """Find the 1.5s window with the highest RMS energy"""
        if len(y) <= self.samples_per_window:
            return np.pad(y, (0, self.samples_per_window - len(y)), mode='constant')
        
        # Calculate RMS in windows to find the 'event'
        hop = self.samples_per_window // 4
        best_rms = -1
        best_start = 0
        
        for start in range(0, len(y) - self.samples_per_window + 1, hop):
            window = y[start:start + self.samples_per_window]
            rms = np.sqrt(np.mean(window**2))
            if rms > best_rms:
                best_rms = rms
                best_start = start
                
        return y[best_start:best_start + self.samples_per_window]

    def extract_features(self, audio_path=None, raw_audio=None, augment=False, return_audio=False):
        if audio_path is not None:
            # Load from file
            y, sr = sf.read(audio_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            # Intelligent Crop (Max energy window)
            y = self._get_max_energy_window(y)
        elif raw_audio is not None:
            # Use raw audio directly (assumed to be 1.5s / 16kHz already)
            y = raw_audio
        else:
            raise ValueError("Either audio_path or raw_audio must be provided")

        # 1.5 Apply DC Notch Filter (Hardware Sync)
        y = self._apply_dc_notch_filter(y)
            
        # Apply MLTK-style Augmentations
        if augment:
            y = self.apply_augmentations(y)
            
        # Feature Generator (Log-Mel)
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, 
                            win_length=self.win_length, window='hann')
        
        mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
                                        fmin=100.0, fmax=7500.0)
        
        mel_spec = np.dot(mel_basis, np.abs(stft)**2)
        
        # Log Scale (MLTK uses 40dB range typically)
        mel_log = librosa.power_to_db(mel_spec, ref=np.max, top_db=40.0)
        
        # Ensure shape is (64, 148)
        if mel_log.shape[1] > 148:
            mel_log = mel_log[:, :148]
        elif mel_log.shape[1] < 148:
            mel_log = np.pad(mel_log, ((0,0), (0, 148 - mel_log.shape[1])), mode='constant')
            
        if return_audio:
            return mel_log, y
        return mel_log

if __name__ == '__main__':
    prep = AFGPreprocessor()
    print("✓ AFG Preprocessor Initialized (16kHz / 64-ch)")
