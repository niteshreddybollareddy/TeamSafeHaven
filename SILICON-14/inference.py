"""
SILICON-14: LIVE INFERENCE ENGINE (Non-Sliding 1.5s Mode)
- Captures 1.5s pulses from the microphone.
- Processes the chunk as a single window (True hardware simulation).
- Minimal latency between captures.
"""

import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
import config
from preprocess import ProPreprocessor
from model import create_model

class LiveInference:
    def __init__(self):
        print("\n" + "="*60)
        print("      SILICON-14: LIVE INFERENCE (1.5s FIXED WINDOW)")
        print("="*60)
        
        # 1. Load Model
        self.model_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model weight file not found at {self.model_path}")
            
        self.model = create_model(num_classes=config.NUM_CLASSES)
        _ = self.model(np.zeros((1, 64, 148, 1)))  # Build graph
        self.model.load_weights(self.model_path)
        print("✅ Model loaded successfully.")
        
        # 2. Setup Preprocessor
        self.prep = ProPreprocessor()
        
        # 3. Audio Setup
        self.chunk_size = 1024
        self.sample_rate = config.SAMPLE_RATE_HZ
        self.buffer_size = config.SAMPLE_LENGTH  # 1.5s = 24000 samples
        self.p = pyaudio.PyAudio()
        
        # Determine Input Device
        self.device_info = self.p.get_default_input_device_info()
        print(f"🎤 Using Microphone: {self.device_info['name']}")
        
    def listen_and_predict(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("\n🚀 LISTENING... (Speak now!)")
        print("-" * 60)
        print(f"{'TIMESTAMP':<15} | {'PREDICTION':<12} | {'CONFIDENCE':<10}")
        print("-" * 60)
        
        try:
            while True:
                # Capture exactly 1.5 seconds of audio
                frames = []
                for _ in range(0, int(self.sample_rate / self.chunk_size * config.SAMPLE_LENGTH_MS / 1000)):
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                
                audio_chunk = np.concatenate(frames)
                
                # 1. Hardware-Style Preprocessing
                # We need to manually call the steps or bypass the file-loading of extract_features
                # Since we have the raw audio, let's use the prep logic:
                
                # DC Notch
                y = self.prep._apply_dc_notch_filter(audio_chunk)
                
                # Loudness Normalization
                y = self.prep.normalize_audio_loudness(y, target_rms=0.12)
                
                # Warm-up padding (Hardware NR simulation)
                noise_warmup = np.random.normal(0, 0.0008, config.WARMUP_SAMPLES)
                y_for_stft = np.concatenate([noise_warmup, y])
                
                # Extraction
                import librosa
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
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
                feat = mel_spec[:, -config.N_TIME_FRAMES:]
                
                # 2. Predict
                feat_input = feat[np.newaxis, ..., np.newaxis]
                probs = self.model.predict(feat_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                label = config.CLASS_NAMES[pred_idx]
                
                # 3. Output
                timestamp = time.strftime("%H:%M:%S")
                if confidence > config.DETECTION_THRESHOLD:
                    print(f"\033[92m{timestamp:<15} | {label:<12} | {confidence:.2%}\033[0m")
                else:
                    print(f"{timestamp:<15} | {'Unknown':<12} | {confidence:.2%}")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopped by User.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()

if __name__ == '__main__':
    engine = LiveInference()
    engine.listen_and_predict()
