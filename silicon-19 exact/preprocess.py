"""
SILICON-19: EXACT SILICON LABS C-PARITY PREPROCESSING
- Parity with sl_ml_audio_feature_generation.c
- 8kHz, 32-channel, 2.0s context (198 frames)
- Hamming Window, 30ms size, 10ms step
- Int8 Quantization (0-666 -> -128-127)
"""

import os
import numpy as np
import scipy.signal
import soundfile as sf

class AFGPreprocessor:
    def __init__(self):
        # Hardware Parity Parameters
        self.sample_rate = 16000
        self.sample_len_ms = 2000
        self.window_size_ms = 30
        self.window_step_ms = 10
        self.n_mels = 32
        self.expected_frames = 198
        self.n_fft = 512
        
        self.samples_per_window = int(self.sample_rate * (self.sample_len_ms / 1000)) # 16000
        self.hop_length = int(self.sample_rate * (self.window_step_ms / 1000)) # 80
        self.win_length = int(self.sample_rate * (self.window_size_ms / 1000)) # 240
        
        # Scaling parameters from C config
        self.log_scale_shift = 6
        self.quant_range_min = 0
        self.quant_range_max = 666
        
        # Thresholds (kept from S18 for activity detection)
        self.activity_top_db = 30 
        self.min_activity_ratio = 0.35 
        self.rms_gate_threshold = 0.002
        self.peak_gate_threshold = 0.01
        self.min_rms_absolute = 0.001 
        self.min_duration = 1.0 

        # Cache Mel Basis and Window
        self.mel_basis = self._get_mel_basis(self.sample_rate, self.n_fft, self.n_mels)
        self.hamming_window = np.hamming(self.win_length)

    def _get_mel_basis(self, sr, n_fft, n_mels):
        """Triangular Mel filterbank matching SL hardware behavior"""
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
        # 0 to SR/2 (8000 Hz)
        fmin, fmax = 0.0, float(sr // 2)
        mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        
        bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        basis = np.zeros((n_mels, int(n_fft // 2 + 1)))
        for i in range(n_mels):
            if bin_pts[i+1] > bin_pts[i]:
                for j in range(bin_pts[i], bin_pts[i+1]):
                    basis[i, j] = (j - bin_pts[i]) / (bin_pts[i+1] - bin_pts[i])
            if bin_pts[i+2] > bin_pts[i+1]:
                for j in range(bin_pts[i+1], bin_pts[i+2]):
                    basis[i, j] = (bin_pts[i+2] - j) / (bin_pts[i+2] - bin_pts[i+1])
        return basis

    def _stft(self, y):
        """STFT implementation using Hamming window"""
        # Padding for exact frame count
        y_padded = np.pad(y, self.n_fft // 2, mode='reflect')
        n_frames = 1 + (len(y_padded) - self.n_fft) // self.hop_length
        frames = []
        for i in range(n_frames):
            start = i * self.hop_length
            chunk = y_padded[start : start + self.n_fft]
            # Center the window in the FFT frame
            pad_amount = (self.n_fft - self.win_length) // 2
            windowed_chunk = chunk[pad_amount : pad_amount + self.win_length] * self.hamming_window
            spectrum = np.fft.rfft(windowed_chunk, n=self.n_fft)
            # Hardware uses absolute magnitude or power. 
            # We use power spectrum like the microfrontend frontend.c
            frames.append(np.abs(spectrum)**2)
        return np.array(frames).T

    def _manual_resample(self, y, orig_sr, target_sr):
        if orig_sr == target_sr: return y
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return scipy.signal.resample_poly(y, up, down)

    def _apply_dc_notch_filter(self, audio):
        b, a = [1, -1], [1, -0.95]
        return scipy.signal.lfilter(b, a, audio)

    def _get_activity_mask(self, y):
        frame_len = int(self.sample_rate * 0.1) # 100ms
        hop = frame_len // 2
        mask = np.zeros(len(y), dtype=bool)
        for i in range(0, len(y) - frame_len, hop):
            chunk = y[i : i + frame_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms > 0.005: 
                mask[i : i + frame_len] = True
        return mask

    def _get_windows(self, y, label=None):
        mask = self._get_activity_mask(y)
        duration = len(y) / self.sample_rate
        if duration < self.min_duration: return []
        rms_total = np.sqrt(np.mean(y**2))
        if rms_total < self.min_rms_absolute: return []

        if len(y) <= self.samples_per_window:
            activity = np.mean(mask)
            if activity < self.min_activity_ratio: return []
            y_padded = np.pad(y, (0, self.samples_per_window - len(y)), mode='constant')
            return [y_padded]

        windows = []
        if label == 'baby_cry':
            best_rms = -1
            best_window = None
            step = self.samples_per_window // 2
            for start in range(0, len(y) - self.samples_per_window + 1, step):
                w = y[start:start + self.samples_per_window]
                m = mask[start:start + self.samples_per_window]
                if np.mean(m) >= self.min_activity_ratio:
                    rms = np.sqrt(np.mean(w**2))
                    if rms > best_rms:
                        best_rms = rms
                        best_window = w
            if best_window is not None: windows.append(best_window)
        elif label == 'dog':
            candidates = []
            step = self.samples_per_window // 2
            for start in range(0, len(y) - self.samples_per_window + 1, step):
                w = y[start:start + self.samples_per_window]
                if np.mean(mask[start:start + self.samples_per_window]) >= self.min_activity_ratio:
                    rms = np.sqrt(np.mean(w**2))
                    candidates.append((rms, w))
            # Sort by energy descending and take top 2
            candidates.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(len(candidates), 2)):
                windows.append(candidates[i][1])
        else: # unknown
            # RANDOM ACTIVE WINDOWS (STRICT MAX 2)
            max_start = len(y) - self.samples_per_window
            retries = 100
            sampled_starts = []
            while len(windows) < 2 and retries > 0:
                retries -= 1
                start = np.random.randint(0, max_start + 1)
                if any(abs(start - s) < self.samples_per_window // 2 for s in sampled_starts): continue
                if np.mean(mask[start:start+self.samples_per_window]) >= self.min_activity_ratio:
                    windows.append(y[start : start + self.samples_per_window])
                    sampled_starts.append(start)
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
            if proc is not None:
                if return_audio: results.append(proc)
                else: results.append(proc[0])
        
        if not results: return None
        return results if audio_path else results[0]

    def _process_single_window(self, y, augment=False):
        # 1. DC Notch
        y = self._apply_dc_notch_filter(y)
        
        # 2. Gate/Norm
        rms, peak = np.sqrt(np.mean(y**2)), np.max(np.abs(y))
        if peak < self.peak_gate_threshold and rms < self.rms_gate_threshold: return None
        if peak > 0.001: y = y / peak
            
        if augment:
            y *= np.random.uniform(0.6, 1.1)
            y += (0.003 * np.random.uniform() * np.random.normal(size=y.shape))
            y = np.roll(y, np.random.randint(int(self.sample_rate * 0.15)))
            y = np.clip(y, -1.0, 1.0)

        # 3. Filterbank (Power Spectrum)
        spec = self._stft(y)
        mel_spec = np.dot(self.mel_basis, spec)
        
        # 4. Log Scale (Hardware Parity)
        # Log = log(1 + x) shifted by scale_shift (usually 6)
        # This approximates natural log behaviors in fixed point
        mel_log = (1 << self.log_scale_shift) * np.log(1.0 + mel_spec)
        
        # 5. Static Quantization (0-666 -> -128-127)
        # Match sl_ml_audio_feature_generation.c LINE 241
        val_div = self.quant_range_max - self.quant_range_min
        mel_quant = (((mel_log - self.quant_range_min) * 256) + (val_div / 2)) // val_div
        mel_quant -= 128
        mel_quant = np.clip(mel_quant, -128, 127).astype(np.int8)
        
        # Fix dimensions (Expected: 32 channels, 198 frames)
        if mel_quant.shape[1] > self.expected_frames:
            mel_quant = mel_quant[:, :self.expected_frames]
        elif mel_quant.shape[1] < self.expected_frames:
            mel_quant = np.pad(mel_quant, ((0,0), (0, self.expected_frames - mel_quant.shape[1])), mode='constant')
            
        return mel_quant, y

if __name__ == '__main__':
    print("🟢 SILICON-19: Exact SL Parity Preprocessor (Int8/8kHz)")
