"""
SILICON-16: PRODUCTION LIVE INFERENCE
- Implements Temporal Consistency (3-Window Streak Rule)
- xG26 Hardware Emulation (Activity Gates + DC Notch)
"""

import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
from preprocess import AFGPreprocessor
from model import create_binary_model

class ProductionInference:
    def __init__(self):
        print("\n" + "="*60)
        print("SILICON-16: PRODUCTION LIVE MONITOR")
        print("="*60)
        
        self.classes = ['baby_cry', 'unknown']
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_binary_model.h5')
        
        # 1. Load Model
        self.model = create_binary_model(input_shape=(64, 198, 1), num_classes=2)
        _ = self.model(np.zeros((1, 64, 198, 1))) 
        self.model.load_weights(self.model_path)
        print("Model Loaded.")
        
        # 2. Setup
        self.prep = AFGPreprocessor()
        self.chunk_size = 4000 # ~250ms chunks
        self.sample_rate = 16000
        self.p = pyaudio.PyAudio()
        
        # 3. State Machine Params
        self.threshold = 0.70
        self.streak_required = 3
        self.current_streak = 0
        
    def listen(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # 2.0s rolling buffer
        buffer = np.zeros(32000, dtype=np.float32)
        
        print("\nmonitoring room for sustained infant patterns...")
        print("-" * 60)
        
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                
                buffer = np.roll(buffer, -len(new_samples))
                buffer[-len(new_samples):] = new_samples
                
                # 1. Preprocess
                feat = self.prep.extract_features(raw_audio=buffer)
                
                if feat is None:
                    # File/Window rejected by Activity/RMS Gate
                    self.current_streak = 0
                    continue
                
                # 2. Inference
                probs = self.model.predict(feat[np.newaxis, ..., np.newaxis], verbose=0)[0]
                cry_conf = probs[0]
                
                # 3. Temporal State Engine
                if cry_conf >= self.threshold:
                    self.current_streak += 1
                else:
                    self.current_streak = 0
                
                # 4. Feedback Logic
                timestamp = time.strftime("%H:%M:%S")
                
                if self.current_streak >= self.streak_required:
                    # SUSTAINED DETECTION
                    meter = "█" * 10
                    print(f"\033[92m{timestamp:<15} | BABY CRYING  | Streak: {self.current_streak} {meter}\033[0m")
                elif self.current_streak > 0:
                    # POTENTIAL START (Building streak)
                    meter = "░" * (self.current_streak * 3)
                    print(f"{timestamp:<15} | ... pattern?    | Streak: {self.current_streak} {meter}")
                else:
                    # No active pattern
                    pass
                    
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()

if __name__ == '__main__':
    monitor = ProductionInference()
    monitor.listen()
