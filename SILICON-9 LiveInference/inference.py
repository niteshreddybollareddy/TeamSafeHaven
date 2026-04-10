"""
SILICON-9 LiveInference Engine
Binary Baby Cry Detection using CryNet-Micro-v2

Input: Audio file (WAV, MP3, OGG, WebM)
Output: Classification (Baby Cry vs Unknown) + confidence
"""
import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf


class AudioPreprocessor:
    """
    MLTK-compatible audio preprocessing for SILICON-9
    Extracts mel-spectrograms matching training configuration
    """

    def __init__(self, config):
        self.sample_rate = config['sample_rate_hz']
        self.sample_length_ms = config['sample_length_ms']
        self.sample_length = int((self.sample_length_ms * self.sample_rate) / 1000)

        self.window_size_ms = config['window_size_ms']
        self.window_step_ms = config['window_step_ms']
        self.n_fft = config['fft_length']

        self.n_mels = config['filterbank_n_channels']
        self.fmin = config['filterbank_lower_band_limit']
        self.fmax = config['filterbank_upper_band_limit']

        self.window_size_length = int((self.window_size_ms * self.sample_rate) / 1000)
        self.window_step_length = int((self.window_step_ms * self.sample_rate) / 1000)

        # Normalization parameters from NEW training (updated)
        self.mean = -31.3581
        self.std = 12.7735

    def _apply_dc_notch_filter(self, audio, coefficient=0.95):
        """DC notch filter"""
        import scipy.signal
        b = [1, -1]
        a = [1, -coefficient]
        return scipy.signal.lfilter(b, a, audio)

    def _select_max_energy_window(self, audio, target_length):
        """Energy-based windowing - PROFESSOR FIX"""
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')

        # Compute RMS energy
        frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512).squeeze()
        max_energy_frame = np.argmax(frame_energy)
        center_sample = max_energy_frame * 512

        # Safe clamping
        start_idx = center_sample - target_length // 2
        start_idx = max(0, min(start_idx, len(audio) - target_length))
        return audio[start_idx:start_idx + target_length]

    def load_and_preprocess(self, audio_path):
        """UPDATED with all professor fixes"""
        # Load audio
        audio, orig_sr = sf.read(audio_path, always_2d=False)

        # Stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

        # DC notch filter
        audio = self._apply_dc_notch_filter(audio)

        # PROFESSOR FIX 1: Energy window BEFORE silence trimming
        if len(audio) < self.sample_length:
            audio = np.pad(audio, (0, self.sample_length - len(audio)), mode='constant')
        else:
            audio = self._select_max_energy_window(audio, self.sample_length)

        # Light silence trimming AFTER windowing
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=40)
        if len(audio_trimmed) < self.sample_length:
            audio = np.pad(audio_trimmed, (0, self.sample_length - len(audio_trimmed)), mode='constant')
        else:
            audio = audio_trimmed[:self.sample_length]

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)

        return mel_spec

    def _apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter to boost high frequencies"""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def extract_mel_spectrogram(self, audio):
        """
        Extract mel-spectrogram using librosa (matches training preprocessing)

        Args:
            audio: Audio samples (numpy array)

        Returns:
            mel_spec: Normalized log mel-spectrogram, shape (64, 98)
        """
        # Apply pre-emphasis (CRITICAL - was missing!)
        audio = self._apply_preemphasis(audio)

        # Compute STFT with center=True to match training
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.window_step_length,
            win_length=self.window_size_length,
            window='hann',
            center=True,  # Match training preprocessing
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
            htk=False  # Slaney-style mel
        )

        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        # Normalize with training statistics (CRITICAL - was missing!)
        mel_spec_normalized = (mel_spec_db - self.mean) / (self.std + 1e-8)

        # Ensure correct time dimension (98 frames for 1.5s audio)
        expected_frames = 98
        if mel_spec_normalized.shape[1] != expected_frames:
            if mel_spec_normalized.shape[1] > expected_frames:
                # Trim excess frames
                mel_spec_normalized = mel_spec_normalized[:, :expected_frames]
            else:
                # Pad if too short
                pad_width = expected_frames - mel_spec_normalized.shape[1]
                mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

        return mel_spec_normalized.astype(np.float32)


class BabyCryClassifier:
    """
    CryNet-Micro-v2 inference engine
    Binary classification: Baby Cry vs Unknown
    """

    def __init__(self, model_path, config, live_threshold=0.65):
        """
        Initialize classifier with temporal voting

        Args:
            model_path: Path to TFLite model
            config: Configuration dictionary
            live_threshold: Confidence threshold for live inference (default: 0.65)
        """
        self.config = config
        self.class_names = config['class_names']

        # Live inference threshold (PROFESSOR FIX: balanced for real-world detection)
        self.live_threshold = live_threshold

        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(config)

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"✓ Model loaded: {model_path}")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
        print(f"  Classes: {self.class_names}")
        print(f"  Live threshold: {live_threshold:.0%} (raised for production)")

    def _predict_single_window(self, mel_spec):
        """Run inference on single window"""
        input_tensor = mel_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]

    def predict(self, audio_path, use_temporal_voting=True):
        """
        Classify audio file with optional temporal voting

        Args:
            audio_path: Path to audio file
            use_temporal_voting: Enable sliding window + voting (default: True)

        Returns:
            dict with:
                - predicted_class: 'Baby Cry' or 'Unknown'
                - confidence: float (0-100)
                - probabilities: dict of class probabilities
                - windows_voted: number of windows analyzed (if voting enabled)
        """
        # Load full audio
        import soundfile as sf
        audio, sr = sf.read(audio_path, always_2d=False)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.config['sample_rate_hz']:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sample_rate_hz'])

        sample_rate = self.config['sample_rate_hz']
        window_length_samples = int(self.config['sample_length_ms'] * sample_rate / 1000)

        # PROFESSOR FIX: Temporal voting with sliding windows
        if use_temporal_voting and len(audio) > window_length_samples:
            hop_samples = sample_rate // 2  # 500ms hop
            baby_votes = []
            all_probs = []

            # Sliding window
            for start in range(0, len(audio) - window_length_samples + 1, hop_samples):
                window = audio[start:start + window_length_samples]

                # Preprocess window (pass as temp file or in-memory)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, window, sample_rate)
                    mel_spec = self.preprocessor.load_and_preprocess(tmp.name)
                    os.remove(tmp.name)

                # Predict
                output = self._predict_single_window(mel_spec)
                baby_prob = float(output[0])
                all_probs.append(baby_prob)

                # Vote if above threshold
                if baby_prob >= self.live_threshold:
                    baby_votes.append(1)
                else:
                    baby_votes.append(0)

            # Check for consecutive votes (PROFESSOR FIX)
            min_consecutive = 3
            max_consecutive = 0
            current_consecutive = 0

            for vote in baby_votes:
                if vote == 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            # Decision: Need 3+ consecutive high-confidence windows
            if max_consecutive >= min_consecutive:
                predicted_class = 'Baby Cry'
                confidence = max(all_probs) * 100  # Use highest confidence
            else:
                predicted_class = 'Unknown'
                confidence = (1 - max(all_probs)) * 100 if all_probs else 50.0

            result = {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'Baby Cry': round(max(all_probs) * 100, 2),
                    'Unknown': round((1 - max(all_probs)) * 100, 2)
                },
                'windows_voted': len(baby_votes),
                'consecutive_detections': max_consecutive
            }

        else:
            # Single window prediction (offline mode)
            mel_spec = self.preprocessor.load_and_preprocess(audio_path)
            output = self._predict_single_window(mel_spec)

            predicted_idx = np.argmax(output)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(output[predicted_idx]) * 100

            result = {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'probabilities': {
                    self.class_names[0]: round(float(output[0]) * 100, 2),
                    self.class_names[1]: round(float(output[1]) * 100, 2)
                }
            }

        return result


def get_config():
    """SILICON-9 configuration matching training setup"""
    return {
        # Audio settings
        'sample_rate_hz': 16000,
        'sample_length_ms': 1500,  # 1.5 seconds

        # STFT settings
        'window_size_ms': 25,
        'window_step_ms': 10,
        'fft_length': 512,

        # Mel filterbank settings
        'filterbank_n_channels': 64,
        'filterbank_lower_band_limit': 50.0,
        'filterbank_upper_band_limit': 8000.0,

        # Class names
        'class_names': ['Baby Cry', 'Unknown']
    }


# Test inference
if __name__ == '__main__':
    import sys

    # Config
    config = get_config()

    # Load classifier
    classifier = BabyCryClassifier(
        model_path='models/crynet_micro_v2.tflite',
        config=config
    )

    # Test with audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"\nTesting inference on: {audio_path}")

        result = classifier.predict(audio_path)

        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.2f}%")
        print(f"\nClass Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls:12s}: {prob:6.2f}%")
        print(f"{'='*60}\n")
    else:
        print("\nUsage: python inference.py <audio_file>")
        print("Example: python inference.py test_audio.wav")
