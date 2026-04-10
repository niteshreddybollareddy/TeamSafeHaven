"""
SILICON-20: REAL-TIME INFERENCE (PCEN)
- implements stateful PCEN streaming.
- applies Z-score normalization using saved stats.
- uses V1 for decision logic.
"""

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
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'best_model.h5')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')
        
        # 1. Load Model
        self.model = create_model(input_shape=(64, 198, 1), num_classes=3)
        # Pre-warmup
        _ = self.model(np.zeros((1, 64, 198, 1))) 
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print("SILICON-20 Model Loaded.")
        else:
            print("Warning: Model weights not found. Running with uninitialized model.")
            
        # 2. Load Normalization Stats
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.norm_mean = stats['mean']
            self.norm_std = stats['std']
            print("PCEN Normalization Stats Loaded.")
        else:
            self.norm_mean = 0
            self.norm_std = 1
            print("Warning: Normalization stats not found. Using defaults.")
        
        # 3. Setup Preprocessor
        self.prep = AFGPreprocessor()
        self.chunk_size = 2000 # ~250ms chunks @ 8kHz
        self.sample_rate = 8000
        self.p = pyaudio.PyAudio()
        
        # 4. State Machine Params (Master Policy V1)
        self.thresholds = {
            'baby_cry': 0.65, 
            'dog': 0.90      # Adjusted for PCEN sensitivity
        }
        self.streaks = {'baby_cry': 0.0, 'dog': 0.0}
        self.required_streaks = {'baby_cry': 2, 'dog': 4}
        self.decay_rates = {
            'baby_cry': 1.0,
            'dog': 0.3       # Dog decays slower to tolerate bursty barks
        }
        
    def listen(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # 2.0s rolling buffer (16000 samples @ 8kHz)
        buffer = np.zeros(16000, dtype=np.float32)
        
        
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                
                buffer = np.roll(buffer, -len(new_samples))
                buffer[-len(new_samples):] = new_samples
                
                # 0. HARD ENERGY GATE (Silence Check)
                rms = np.sqrt(np.mean(buffer**2))
                if rms < 0.002: # Silence Threshold
                    self.prep.reset_pcen_state() # Flush background estimate on silence
                    for k in self.streaks: 
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                    continue

                # 0.1 SPEECH REJECTION GATE (Envelope Slope Analysis)
                # Speech has slower/syllabic envelope compared to impulsive dog barks
                frame_05s = buffer[-4000:] # Last 0.5s @ 8kHz
                env = np.sqrt(np.mean(frame_05s.reshape(-1, 200)**2, axis=1)) # 25ms RMS frames
                slope = np.mean(np.diff(env))
                
                if slope < 0.0005:
                    # Likely speech or background noise (non-impulsive)
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                    # We don't 'continue' here for baby cry since cry is also sustained.
                    pass

                # 1. Preprocess (PCEN with is_inference=True for state persistence)
                # This call updates internal prep.pcen_state automatically
                feat = self.prep.extract_features(raw_audio=buffer, is_inference=True)
                
                if feat is None:
                    # Decay streaks on low PCEN-space energy
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                    continue
                
                # 2. Z-Score Normalization (Match Training)
                # Reshape feat to (64, 198, 1) first to match training data shape
                feat_3d = feat[..., np.newaxis]
                
                # Extract scalars from norm stats if they were saved with keepdims=True
                mean_val = self.norm_mean.item() if hasattr(self.norm_mean, 'item') else self.norm_mean
                std_val = self.norm_std.item() if hasattr(self.norm_std, 'item') else self.norm_std
                
                feat_norm = (feat_3d - mean_val) / (std_val + 1e-9)
                
                # 3. Inference
                # feat_norm is (64, 198, 1), we add batch dim to get (1, 64, 198, 1)
                probs = self.model.predict(feat_norm[np.newaxis, ...], verbose=0)[0]
                baby_prob = probs[0]
                dog_prob = probs[1]

                # 4. Master Policy V1 (Refined for Speech Rejection)
                # Dead zone: If nothing is confident, decay
                if max(baby_prob, dog_prob) < 0.7:
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                
                elif baby_prob >= self.thresholds['baby_cry']:
                    self.streaks['baby_cry'] += 1
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                
                # Dog Relative Conf Rule: Dog must beat baby by at least 15% to trigger
                elif dog_prob >= self.thresholds['dog'] and dog_prob > baby_prob + 0.15:
                    self.streaks['dog'] += 1
                    self.streaks['baby_cry'] = max(0, self.streaks['baby_cry'] - self.decay_rates['baby_cry'])
                
                else:
                    # Decay both if neither hit or relative rule failed
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                
                # 5. UI Feedback
                timestamp = time.strftime("%H:%M:%S")
                
                # Format streaks for display (rounded)
                b_strk = round(self.streaks['baby_cry'], 1)
                d_strk = round(self.streaks['dog'], 1)
                
                if self.streaks['baby_cry'] >= self.required_streaks['baby_cry']:
                    meter = "#" * 12
                    print(f"\033[92m{timestamp:<15} | BABY CRYING  | Streak: {b_strk:<5} {meter}\033[0m")
                elif self.streaks['dog'] >= self.required_streaks['dog']:
                    meter = "#" * 12
                    print(f"\033[94m{timestamp:<15} | DOG BARKING  | Streak: {d_strk:<5} {meter}\033[0m")
                elif self.streaks['baby_cry'] > 0:
                    meter = "*" * int(self.streaks['baby_cry'] * 4)
                    print(f"{timestamp:<15} | ... baby?     | Streak: {b_strk:<5} {meter}")
                elif self.streaks['dog'] > 0:
                    meter = "*" * int(self.streaks['dog'] * 2)
                    print(f"{timestamp:<15} | ... dog?      | Streak: {d_strk:<5} {meter}")
                    
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()

if __name__ == '__main__':
    monitor = ProductionInference()
    monitor.listen()
