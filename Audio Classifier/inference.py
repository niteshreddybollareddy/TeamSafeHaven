"""
SILICON-21: REAL-TIME INFERENCE (CryNet)
- Implements stateful PCEN streaming.
- Applies Z-score normalization using saved stats.
- Uses Master Policy V1 for decision logic (Silicon-22 / xG26 Optimized).
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
        self.model_path = os.path.join(self.base_dir, 'best_model.keras')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')
        
        # 1. Load Model
        self.model = create_model(input_shape=(64, 198, 1), num_classes=3)
        # Pre-warmup
        _ = self.model(np.zeros((1, 64, 198, 1))) 
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print("✅ SILICON-22 Model Loaded.")
        else:
            print("⚠️ Warning: Model weights not found. Running with uninitialized model.")
            
        # 2. Load Normalization Stats
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.norm_mean = stats['mean'].squeeze() if hasattr(stats['mean'], 'squeeze') else stats['mean']
            self.norm_std = stats['std'].squeeze() if hasattr(stats['std'], 'squeeze') else stats['std']
            print("✅ PCEN Normalization Stats Loaded.")
        else:
            self.norm_mean = 0
            self.norm_std = 1
            print("⚠️ Warning: Normalization stats not found. Using defaults.")
        
        # 3. Setup Preprocessor
        self.prep = AFGPreprocessor()
        self.chunk_size = 4000 # ~250ms chunks @ 16kHz
        self.sample_rate = 16000
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
        self.baby_suppression_floor = 0.35 # Fix: Dog must not trigger if baby is semi-active
        
    def listen(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # 2.0s rolling buffer (32000 samples @ 16kHz)
        # NOTE: For MCU porting, this buffer must be int16 to fit in RAM
        buffer = np.zeros(32000, dtype=np.float32)
        
        print("\n👂 Monitoring room (SILICON-21: CryNet Architecture + Master Policy V1)")
        print("-" * 75)
        
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
                frame_05s = buffer[-8000:] # Last 0.5s
                env = np.sqrt(np.mean(frame_05s.reshape(-1, 400)**2, axis=1)) # 25ms RMS frames
                slope = np.mean(np.diff(env))
                
                if slope < 0.0005:
                    # Likely speech or background noise (non-impulsive)
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                    pass

                # 1. Preprocess (PCEN with is_inference=True for state persistence)
                feat = self.prep.extract_features(raw_audio=buffer, is_inference=True)

                if feat is None:
                    for k in self.streaks:
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                    continue

                # 2. Force deterministic shape (64, 198)
                feat = np.array(feat)
                feat = np.squeeze(feat) # Remove all dummy dims
                
                try:
                    feat = feat.reshape(64, 198)
                except ValueError:
                    print(f"⚠️ Unexpected feature shape after squeeze: {feat.shape}")
                    continue

                # 3. Add channel dim -> (64, 198, 1)
                feat = feat[:, :, np.newaxis]

                # 4. Normalize (Z-Score)
                feat = (feat - self.norm_mean) / (self.norm_std + 1e-9)

                # 5. Add batch dim -> (1, 64, 198, 1)
                feat_input = feat[np.newaxis, ...]

                # 6. Predict
                probs = self.model.predict(feat_input, verbose=0)[0]
                baby_prob = probs[0]
                dog_prob = probs[1]

                # 4. Master Policy V1 (CryNet Refined)
                # Dead zone: If nothing is confident, decay
                if max(baby_prob, dog_prob) < 0.7:
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                
                elif baby_prob >= self.thresholds['baby_cry']:
                    self.streaks['baby_cry'] += 1
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                
                # Dog Prediction: Requires confidence, dominance margin, AND baby suppression
                elif (dog_prob >= self.thresholds['dog'] and 
                      (dog_prob - baby_prob) > 0.15 and 
                      baby_prob < self.baby_suppression_floor):
                    self.streaks['dog'] += 1
                    self.streaks['baby_cry'] = max(0, self.streaks['baby_cry'] - self.decay_rates['baby_cry'])
                
                else:
                    # Decay both if no rule satisfied
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
