"""
SILICON-15: MLTK-STYLE LIVE INFERENCE
Binary Classifier: baby_cry vs unknown
Pulse Mode: 1.5s fixed windows
"""

import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
from preprocess import AFGPreprocessor
from model import create_binary_model

class BinaryInference:
    def __init__(self):
        print("\n" + "="*60)
        print("      SILICON-15: BINARY LIVE INFERENCE (MLTK-STYLE)")
        print("="*60)
        
        self.classes = ['baby_cry', 'unknown']
        
        # 1. Load Model
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_binary_model.h5')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model weight file not found at {self.model_path}")
            
        self.model = create_binary_model(input_shape=(64, 148, 1), num_classes=2)
        # Build graph
        _ = self.model(np.zeros((1, 64, 148, 1))) 
        self.model.load_weights(self.model_path)
        print("✅ Binary Model loaded successfully.")
        
        # 2. Setup Preprocessor (Uses exactly the same logic as training)
        self.prep = AFGPreprocessor()
        
        # 3. Audio Setup
        self.chunk_size = 1024
        self.sample_rate = 16000
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
        
        print("\n🚀 LISTENING... (1.5s Pulses)")
        print("-" * 60)
        print(f"{'TIMESTAMP':<15} | {'PREDICTION':<15} | {'CONFIDENCE':<10}")
        print("-" * 60)
        
        # Threshold for detection (MLTK default is usually around 0.85-0.95 for final output)
        threshold = 0.90
        
        try:
            while True:
                # Capture exactly 1.5 seconds of audio (24,000 samples)
                frames = []
                # 1.5s / (1024/16000) ~= 23.4 chunks
                for _ in range(0, int(16000 / self.chunk_size * 1.5)):
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                
                audio_chunk = np.concatenate(frames)
                
                # --- MLTK PREPROCESSING PIPELINE ---
                # Use the exact same logic as training
                feat = self.prep.extract_features(raw_audio=audio_chunk)
                
                # --- INFERENCE ---
                feat_input = feat[np.newaxis, ..., np.newaxis]
                probs = self.model.predict(feat_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                label = self.classes[pred_idx]
                
                # --- OUTPUT ---
                timestamp = time.strftime("%H:%M:%S")
                
                # Only show detection if it's 'baby_cry' and over threshold, otherwise it's 'unknown'
                if label == 'baby_cry' and confidence > threshold:
                    # Highlight in Green
                    print(f"\033[92m{timestamp:<15} | 👶 {label:<12} | {confidence:.2%}\033[0m")
                else:
                    # Generic output
                    print(f"{timestamp:<15} | ❓ unknown      | {confidence:.2%}")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopped by User.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()

if __name__ == '__main__':
    engine = BinaryInference()
    engine.listen_and_predict()
