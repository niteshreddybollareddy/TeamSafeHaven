"""
SILICON-16: STRICT ACTIVITY-AWARE PREPROCESSING (V3)
- 16kHz, 64-channel, 2.0s window (198 frames)
- Log-Scale (40dB range), DC Notch 0.95
- STRICT Filter: Minimum 40% (0.8s) activity required per window.
- Rejects files with insufficient signal to prevent silence-poisoning.
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal

class AFGPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.sample_len_ms = 2000
        self.window_size_ms = 30
        self.window_step_ms = 10
        self.n_mels = 64
        self.expected_frames = 198
        self.samples_per_window = int(self.sample_rate * (self.sample_len_ms / 1000))
        self.n_fft = 512
        self.hop_length = int(self.sample_rate * (self.window_step_ms / 1000))
        self.win_length = int(self.sample_rate * (self.window_size_ms / 1000))
        
        # Thresholds
        self.activity_top_db = 30 
        self.min_activity_ratio = 0.40 # 40% of window must be active
        self.rms_gate_threshold = 0.002
        self.peak_norm_threshold = 0.01

    def _apply_dc_notch_filter(self, audio):
        b, a = [1, -1], [1, -0.95]
        return scipy.signal.lfilter(b, a, audio)

    def _get_activity_mask(self, y):
        """Returns boolean mask of active regions"""
        intervals = librosa.effects.split(y, top_db=self.activity_top_db)
        mask = np.zeros(len(y), dtype=bool)
        for start, end in intervals:
            mask[start:end] = True
        return mask

    def _get_window(self, y, label=None):
        """Strictly finds a 2.0s window with >= 40% activity."""
        mask = self._get_activity_mask(y)
        
        # If total audio is shorter than window, check its total activity
        if len(y) <= self.samples_per_window:
            activity = np.mean(mask)
            if activity < self.min_activity_ratio: return None
            return np.pad(y, (0, self.samples_per_window - len(y)), mode='constant')

        if label == 'baby_cry':
            # FIND MAX ENERGY WINDOW
            best_rms = -1
            best_window = None
            step = self.samples_per_window // 4
            for start in range(0, len(y) - self.samples_per_window + 1, step):
                window = y[start:start + self.samples_per_window]
                w_mask = mask[start:start + self.samples_per_window]
                if np.mean(w_mask) >= self.min_activity_ratio:
                    rms = np.sqrt(np.mean(window**2))
                    if rms > best_rms:
                        best_rms = rms
                        best_window = window
            return best_window # Returns None if no window met the 40% rule
            
        else:
            # RANDOM ACTIVE WINDOW (With 20 retries)
            max_start = len(y) - self.samples_per_window
            for _ in range(20):
                start = np.random.randint(0, max_start + 1)
                if np.mean(mask[start:start + self.samples_per_window]) >= self.min_activity_ratio:
                    return y[start:start + self.samples_per_window]
            return None # Failed to find an active enough random window

    def extract_features(self, audio_path=None, raw_audio=None, label=None, augment=False, return_audio=False):
        if audio_path is not None:
            y, sr = sf.read(audio_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            y = self._get_window(y, label=label)
            if y is None: return None # REJECT FILE
        else:
            y = raw_audio
            if len(y) < self.samples_per_window:
                y = np.pad(y, (0, self.samples_per_window - len(y)), mode='constant')
            elif len(y) > self.samples_per_window:
                y = y[:self.samples_per_window]

        # Standard pipeline
        y = self._apply_dc_notch_filter(y)
        
        # Noise Gate check (pre-norm)
        rms = np.sqrt(np.mean(y**2))
        if rms < self.rms_gate_threshold: return None # REJECT TOO QUIET
        
        if np.max(np.abs(y)) > self.peak_norm_threshold:
            y = y / np.max(np.abs(y))
            
        if augment:
            # Custom gain 0.5-1.2, White noise 0.5%, Time shift
            y = y * np.random.uniform(0.5, 1.2)
            y = y + (0.005 * np.random.uniform() * np.random.normal(size=y.shape))
            y = np.roll(y, np.random.randint(int(self.sample_rate * 0.1)))
            y = np.clip(y, -1.0, 1.0)

        # FBANK
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, 
                            win_length=self.win_length, window='hann')
        mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
                                        fmin=50.0, fmax=8000.0)
        mel_spec = np.dot(mel_basis, np.abs(stft)**2)
        mel_log = librosa.power_to_db(mel_spec, ref=np.max, top_db=40.0)
        
        # Crop/Pad to exact frames
        if mel_log.shape[1] != self.expected_frames:
            if mel_log.shape[1] > self.expected_frames:
                mel_log = mel_log[:, :self.expected_frames]
            else:
                mel_log = np.pad(mel_log, ((0,0), (0, self.expected_frames - mel_log.shape[1])), mode='constant')
            
        if return_audio: return mel_log, y
        return mel_log

if __name__ == '__main__':
    print("🟢 SILICON-16: Strict V3 Preprocessor Initialized (40% Activity Rule)")
