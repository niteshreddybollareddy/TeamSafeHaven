"""
SILICON-22: REAL-TIME INT8 INFERENCE (CryNet)
- Implements stateful PCEN streaming.
- Applies Z-score normalization using saved stats.
- Uses Master Policy V1 for decision logic.
- Optimized for TFLite INT8 Quantized model on xG26 simulation.
"""

import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
from preprocess import AFGPreprocessor

class ProductionInference:
    def __init__(self):
        self.classes = ['baby_cry', 'dog', 'unknown']
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'int8_model.tflite')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')
        
        # 1. Load TFLite Model
        if os.path.exists(self.model_path):
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ SILICON-22 INT8 TFLite Model Loaded: {self.model_path}")
        else:
            print(f"❌ Error: TFLite model not found at {self.model_path}")
            exit(1)
            
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
        buffer = np.zeros(32000, dtype=np.float32)
        
        print("\n👂 Monitoring room (SILICON-22: CryNet INT8 + Master Policy V1)")
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
                frame_05s = buffer[-8000:] # Last 0.5s
                env = np.sqrt(np.mean(frame_05s.reshape(-1, 400)**2, axis=1)) # 25ms RMS frames
                slope = np.mean(np.diff(env))
                
                if slope < 0.0005:
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                    pass

                # 1. Preprocess
                feat = self.prep.extract_features(raw_audio=buffer, is_inference=True)

                if feat is None:
                    for k in self.streaks:
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                    continue

                # 2. Force deterministic shape (64, 198)
                feat = np.array(feat)
                feat = np.squeeze(feat) 
                
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
                feat_input = feat[np.newaxis, ...].astype(np.float32)

                # 6. TFLite INT8 Inference
                # Check if quantization is needed
                input_details = self.input_details[0]
                if input_details['dtype'] == np.int8:
                    scale, zero_point = input_details['quantization']
                    feat_input = (feat_input / scale + zero_point).astype(np.int8)
                
                self.interpreter.set_tensor(input_details['index'], feat_input)
                self.interpreter.invoke()
                
                output_details = self.output_details[0]
                probs = self.interpreter.get_tensor(output_details['index'])[0]
                
                # Dequantize output if it is int8
                if output_details['dtype'] == np.int8:
                    scale, zero_point = output_details['quantization']
                    probs = (probs.astype(np.float32) - zero_point) * scale

                baby_prob = probs[0]
                dog_prob = probs[1]

                # 4. Master Policy V1
                if max(baby_prob, dog_prob) < 0.7:
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                
                elif baby_prob >= self.thresholds['baby_cry']:
                    self.streaks['baby_cry'] += 1
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                
                elif (dog_prob >= self.thresholds['dog'] and 
                      (dog_prob - baby_prob) > 0.15 and 
                      baby_prob < self.baby_suppression_floor):
                    self.streaks['dog'] += 1
                    self.streaks['baby_cry'] = max(0, self.streaks['baby_cry'] - self.decay_rates['baby_cry'])
                
                else:
                    for k in self.streaks: self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                
                # 5. UI Feedback
                timestamp = time.strftime("%H:%M:%S")
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
