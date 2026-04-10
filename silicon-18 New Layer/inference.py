

import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
from preprocess import AFGPreprocessor
from model import create_model

class ProductionInference:
    def __init__(self):
        
        self.classes = ['baby_cry', 'dog', 'unknown']
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.h5')
        
        # 1. Load Model
        self.model = create_model(input_shape=(64, 198, 1), num_classes=3)
        _ = self.model(np.zeros((1, 64, 198, 1))) 
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print(" Model Loaded.")
        else:
            print("Warning: Model weights not found. Running with uninitialized model.")
        
        # 2. Setup
        self.prep = AFGPreprocessor()
        self.chunk_size = 4000 # ~250ms chunks
        self.sample_rate = 16000
        self.p = pyaudio.PyAudio()
        
        # 3. State Machine Params (Master Policy V1)
        self.thresholds = {
            'baby_cry': 0.65, # Target Precision: 100%
            'dog': 0.92      # Raised to prevent pollution
        }
        self.streaks = {'baby_cry': 0, 'dog': 0}
        self.required_streaks = {'baby_cry': 2, 'dog': 4}
        
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
        
        print("\n👂 Monitoring room (Master Policy V1: Priority Baby)")
        print("-" * 60)
        
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                
                buffer = np.roll(buffer, -len(new_samples))
                buffer[-len(new_samples):] = new_samples
                
                # 1. Preprocess (V5 extracts a SINGLE window in inference mode)
                feat = self.prep.extract_features(raw_audio=buffer)
                
                if feat is None:
                    # Decay both on silence
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - 1)
                    continue
                
                # 2. Inference
                probs = self.model.predict(feat[np.newaxis, ..., np.newaxis], verbose=0)[0]
                baby_prob = probs[0]
                dog_prob = probs[1]

                # 3. Priority Decision Logic (Rule: Decay, not hard reset)
                if baby_prob >= self.thresholds['baby_cry']:
                    self.streaks['baby_cry'] += 1
                    self.streaks['dog'] = max(0, self.streaks['dog'] - 1)
                elif dog_prob >= self.thresholds['dog']:
                    self.streaks['dog'] += 1
                    self.streaks['baby_cry'] = max(0, self.streaks['baby_cry'] - 1)
                else:
                    # Decay both if neither hits
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - 1)
                
                # 4. Feedback Logic (Priority Order)
                timestamp = time.strftime("%H:%M:%S")
                
                # Check Baby first
                if self.streaks['baby_cry'] >= self.required_streaks['baby_cry']:
                    meter = "#" * 10
                    print(f"\033[92m{timestamp:<15} | BABY CRYING  | Streak: {self.streaks['baby_cry']} {meter}\033[0m")
                elif self.streaks['dog'] >= self.required_streaks['dog']:
                    meter = "#" * 10
                    print(f"\033[94m{timestamp:<15} | DOG BARKING  | Streak: {self.streaks['dog']} {meter}\033[0m")
                elif self.streaks['baby_cry'] > 0:
                    meter = "*" * (self.streaks['baby_cry'] * 5)
                    print(f"{timestamp:<15} | ... baby_cry?  | Streak: {self.streaks['baby_cry']} {meter}")
                elif self.streaks['dog'] > 0:
                    meter = "*#" * (self.streaks['dog'] * 2)
                    print(f"{timestamp:<15} | ... dog?       | Streak: {self.streaks['dog']} {meter}")
                else:
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
