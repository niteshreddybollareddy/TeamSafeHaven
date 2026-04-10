"""
SILICON-16: LIBROSA-FREE PREPROCESSING (V5)
- All audio functions implemented manually via NumPy/SciPy.
- 16kHz, 64-channel, 2.0s context (198 frames)
- Log-Mel (40dB range), DC Notch 0.95
- Manual Resampling (ResamplePoly), STFT, and Mel Filterbank.
- Class-aware windowing and stateful Activity Detection.
"""

import numpy as np
import scipy.signal
import soundfile as sf

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
        self.min_activity_ratio = 0.35 
        self.rms_gate_threshold = 0.002
        self.peak_gate_threshold = 0.01
        self.min_rms_absolute = 0.001 
        self.min_duration = 1.0 

        # Cache Mel Basis and Window
        self.mel_basis = self._get_mel_basis(self.sample_rate, self.n_fft, self.n_mels)
        self.hann_window = np.hanning(self.win_length)

    def _get_mel_basis(self, sr, n_fft, n_mels):
        """Implements triangular Mel filterbank manually (HTK scale)"""
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
        fmin, fmax = 50.0, 8000.0
        mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        
        # Convert Hz to FFT bin indices
        bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        
        basis = np.zeros((n_mels, int(n_fft // 2 + 1)))
        for i in range(n_mels):
            # Left slope
            if bin_pts[i+1] > bin_pts[i]:
                for j in range(bin_pts[i], bin_pts[i+1]):
                    basis[i, j] = (j - bin_pts[i]) / (bin_pts[i+1] - bin_pts[i])
            # Right slope
            if bin_pts[i+2] > bin_pts[i+1]:
                for j in range(bin_pts[i+1], bin_pts[i+2]):
                    basis[i, j] = (bin_pts[i+2] - j) / (bin_pts[i+2] - bin_pts[i+1])
                
            # Area Normalization (Slaney-style)
            enorm = 2.0 / (hz_pts[i+2] - hz_pts[i])
            basis[i] *= enorm
            
        return basis

    def _stft(self, y):
        """Manual STFT implementation matching parameters"""
        # Padding for centering (reflect)
        y_padded = np.pad(y, self.n_fft // 2, mode='reflect')
        n_frames = 1 + (len(y_padded) - self.n_fft) // self.hop_length
        
        frames = []
        for i in range(n_frames):
            start = i * self.hop_length
            chunk = y_padded[start : start + self.n_fft]
            # Center the win_length window within n_fft buffer
            pad_amount = (self.n_fft - self.win_length) // 2
            windowed_chunk = chunk[pad_amount : pad_amount + self.win_length] * self.hann_window
            # FFT of chunk
            spectrum = np.fft.rfft(windowed_chunk, n=self.n_fft)
            frames.append(np.abs(spectrum)**2)
            
        return np.array(frames).T

    def _manual_resample(self, y, orig_sr, target_sr):
        """Manual resampling using scipy.signal.resample_poly"""
        if orig_sr == target_sr: return y
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return scipy.signal.resample_poly(y, up, down)

    def _apply_dc_notch_filter(self, audio):
        b, a = [1, -1], [1, -0.95]
        return scipy.signal.lfilter(b, a, audio)

    def _get_activity_mask(self, y):
        """Manual activity mask using 100ms RMS frames"""
        frame_len = int(self.sample_rate * 0.1) # 100ms
        hop = frame_len // 2
        
        mask = np.zeros(len(y), dtype=bool)
        for i in range(0, len(y) - frame_len, hop):
            chunk = y[i : i + frame_len]
            rms = np.sqrt(np.mean(chunk**2))
            # ~30dB below peak threshold relative to 1.0
            if rms > 0.005: 
                mask[i : i + frame_len] = True
        return mask

    def _get_windows(self, y, label=None):
        """Extracts windows based on class strategy"""
        mask = self._get_activity_mask(y)
        if len(y) / self.sample_rate < self.min_duration: return []
        if np.sqrt(np.mean(y**2)) < self.min_rms_absolute: return []

        if len(y) <= self.samples_per_window:
            if np.mean(mask) < self.min_activity_ratio: return []
            return [np.pad(y, (0, self.samples_per_window-len(y)), mode='constant')]

        windows = []
        step = self.samples_per_window // 2
        candidates = []

        for start in range(0, len(y) - self.samples_per_window + 1, step):
            w = y[start : start + self.samples_per_window]
            m = mask[start : start + self.samples_per_window]
            if np.mean(m) >= self.min_activity_ratio:
                rms = np.sqrt(np.mean(w**2))
                candidates.append((rms, start, w))

        if not candidates: return []

        if label == 'baby_cry':
            candidates.sort(key=lambda x: x[0], reverse=True)
            windows.append(candidates[0][2])
        else: # unknown
            # RANDOM ACTIVE WINDOWS (STRICT MAX 3)
            np.random.shuffle(candidates)
            for c in candidates[:3]: windows.append(c[2])
            
        return windows

    def extract_features(self, audio_path=None, raw_audio=None, label=None, augment=False, return_audio=False):
        if audio_path:
            y, sr = sf.read(audio_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            y = self._manual_resample(y, sr, self.sample_rate)
            audios = self._get_windows(y, label)
            if not audios: return None
        else:
            y = raw_audio
            if len(y) < self.samples_per_window:
                y = np.pad(y, (0, self.samples_per_window - len(y)), mode='constant')
            audios = [y[:self.samples_per_window]]

        results = []
        for a in audios:
            proc = self._process_single_window(a, augment)
            if proc:
                if return_audio: results.append(proc)
                else: results.append(proc[0])
        
        if not results: return None
        return results if audio_path else results[0]

    def _process_single_window(self, y, augment=False):
        # 1. DC Notch
        y = self._apply_dc_notch_filter(y)
        
        # 2. Gate
        rms, peak = np.sqrt(np.mean(y**2)), np.max(np.abs(y))
        if peak < self.peak_gate_threshold and rms < self.rms_gate_threshold: return None
        
        # 3. Norm
        if peak > 0.001: y = y / peak
            
        if augment:
            y *= np.random.uniform(0.6, 1.1)
            y += (0.003 * np.random.uniform() * np.random.normal(size=y.shape))
            y = np.roll(y, np.random.randint(int(self.sample_rate * 0.15)))
            y = np.clip(y, -1.0, 1.0)

        # 4. Filterbank
        spec = self._stft(y)
        mel_spec = np.dot(self.mel_basis, spec)
        # Power to DB (40dB range)
        mel_log = 10.0 * np.log10(np.maximum(1e-10, mel_spec))
        ref = np.max(mel_log)
        mel_log = np.maximum(mel_log, ref - 40.0)
        
        # Fix dimensions
        if mel_log.shape[1] > self.expected_frames:
            mel_log = mel_log[:, :self.expected_frames]
        elif mel_log.shape[1] < self.expected_frames:
            mel_log = np.pad(mel_log, ((0,0), (0, self.expected_frames - mel_log.shape[1])), mode='constant')
            
        return mel_log, y

if __name__ == '__main__':
    print("SILICON-16 V5")
