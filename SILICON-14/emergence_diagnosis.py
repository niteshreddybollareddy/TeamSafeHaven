"""
EMERGENCY DIAGNOSIS: Why Both Models Fail on Real Data
This script will identify the EXACT problem.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
import config
from preprocess import ProPreprocessor
from model import create_model

class EmergencyDiagnostic:
    def __init__(self):
        self.prep = ProPreprocessor()
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        model_path = os.path.join(config.MODEL_DIR, 'best_demo_model.h5')
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return False
        
        self.model = create_model(num_classes=config.NUM_CLASSES)
        _ = self.model(np.zeros((1, 64, 148, 1)))
        self.model.load_weights(model_path)
        print(f"✅ Model loaded from {model_path}\n")
        return True
    
    def test_1_model_on_training_samples(self):
        """TEST 1: Does model correctly predict its own training data?"""
        print("="*70)
        print("TEST 1: MODEL SANITY CHECK - Predict Training Samples")
        print("="*70)
        
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.PROCESSED_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️  {class_name} directory not found")
                continue
            
            # Load 5 random preprocessed features
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('_clean.npy')][:5]
            
            correct = 0
            for npy_file in npy_files:
                feat = np.load(os.path.join(class_dir, npy_file))
                feat_input = feat[np.newaxis, ..., np.newaxis]
                
                probs = self.model.predict(feat_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_label = config.CLASS_NAMES[pred_idx]
                conf = probs[pred_idx]
                
                if pred_label == class_name:
                    correct += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"  {status} {class_name:12} -> {pred_label:12} ({conf:.3f})")
            
            print(f"  Accuracy: {correct}/{len(npy_files)}\n")
        
        print("VERDICT:")
        print("  - If accuracy < 80%: Model file is corrupted or didn't load")
        print("  - If accuracy > 95%: Model is fine, problem is elsewhere")
        print()
    
    def test_2_compare_train_vs_test_features(self):
        """TEST 2: Are test features dramatically different from training features?"""
        print("="*70)
        print("TEST 2: FEATURE DISTRIBUTION COMPARISON")
        print("="*70)
        
        # Load training features
        train_features = []
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.PROCESSED_DIR, class_name)
            if not os.path.exists(class_dir):
                continue
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('_clean.npy')][:20]
            for f in npy_files:
                feat = np.load(os.path.join(class_dir, f))
                train_features.append(feat)
        
        train_features = np.array(train_features)
        
        print("TRAINING DATA STATISTICS:")
        print(f"  Shape: {train_features.shape}")
        print(f"  Mean: {train_features.mean():.3f}")
        print(f"  Std: {train_features.std():.3f}")
        print(f"  Min: {train_features.min():.3f}")
        print(f"  Max: {train_features.max():.3f}")
        print(f"  25th percentile: {np.percentile(train_features, 25):.3f}")
        print(f"  50th percentile: {np.percentile(train_features, 50):.3f}")
        print(f"  75th percentile: {np.percentile(train_features, 75):.3f}")
        
        # Load test features
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
        if not os.path.exists(test_dir):
            print("\n⚠️  No test directory found. Skipping test feature comparison.")
            return
        
        test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3'))][:10]
        if not test_files:
            print("\n⚠️  No test audio files found.")
            return
        
        test_features = []
        print(f"\nTEST DATA STATISTICS (from {len(test_files)} files):")
        
        for test_file in test_files:
            try:
                feat = self.prep.extract_features(os.path.join(test_dir, test_file), augment=False)
                test_features.append(feat)
            except Exception as e:
                print(f"  Error processing {test_file}: {e}")
        
        if test_features:
            test_features = np.array(test_features)
            print(f"  Shape: {test_features.shape}")
            print(f"  Mean: {test_features.mean():.3f}")
            print(f"  Std: {test_features.std():.3f}")
            print(f"  Min: {test_features.min():.3f}")
            print(f"  Max: {test_features.max():.3f}")
            print(f"  25th percentile: {np.percentile(test_features, 25):.3f}")
            print(f"  50th percentile: {np.percentile(test_features, 50):.3f}")
            print(f"  75th percentile: {np.percentile(test_features, 75):.3f}")
            
            # Compare
            mean_diff = abs(train_features.mean() - test_features.mean())
            std_diff = abs(train_features.std() - test_features.std())
            
            print(f"\nDIFFERENCE ANALYSIS:")
            print(f"  Mean difference: {mean_diff:.3f}")
            print(f"  Std difference: {std_diff:.3f}")
            
            if mean_diff > 10:
                print("  ❌ CRITICAL: Feature means are VERY different!")
                print("     This will cause prediction failures.")
            elif mean_diff > 5:
                print("  ⚠️  WARNING: Significant mean difference")
            else:
                print("  ✅ Feature distributions are similar")
        
        print()
    
    def test_3_audio_quality_check(self):
        """TEST 3: Are test audio files actually valid?"""
        print("="*70)
        print("TEST 3: TEST AUDIO FILE QUALITY CHECK")
        print("="*70)
        
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
        if not os.path.exists(test_dir):
            print("⚠️  No test directory found.")
            return
        
        test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
        
        if not test_files:
            print("❌ No audio files found in test directory!")
            return
        
        print(f"Found {len(test_files)} test files. Analyzing first 10...\n")
        
        for test_file in test_files[:10]:
            path = os.path.join(test_dir, test_file)
            print(f"📁 {test_file}")
            
            try:
                # Load with soundfile
                y, sr = sf.read(path, always_2d=False)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                
                duration = len(y) / sr
                energy = np.sqrt(np.mean(y**2))
                peak = np.abs(y).max()
                
                print(f"   Sample Rate: {sr} Hz")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Samples: {len(y)}")
                print(f"   Data Range: [{y.min():.3f}, {y.max():.3f}]")
                print(f"   RMS Energy: {energy:.4f}")
                print(f"   Peak Amplitude: {peak:.4f}")
                
                # Check for problems
                issues = []
                if sr != 16000:
                    issues.append(f"⚠️  Sample rate {sr} Hz (will be resampled)")
                if duration < 0.5:
                    issues.append("⚠️  Very short audio (<0.5s)")
                if energy < 0.001:
                    issues.append("❌ CRITICAL: Audio is nearly silent!")
                if peak > 0.99:
                    issues.append("⚠️  Audio may be clipped")
                if np.all(y == 0):
                    issues.append("❌ CRITICAL: Audio is completely silent!")
                
                if issues:
                    for issue in issues:
                        print(f"   {issue}")
                else:
                    print(f"   ✅ Audio quality looks good")
                
                print()
                
            except Exception as e:
                print(f"   ❌ ERROR loading file: {e}\n")
        
        print("VERDICT:")
        print("  - Silent/low-energy files will be misclassified")
        print("  - Clipped audio may have distorted features")
        print()
    
    def test_4_source_data_vs_test_data(self):
        """TEST 4: Compare raw source audio used in training vs test audio"""
        print("="*70)
        print("TEST 4: TRAINING SOURCE vs TEST AUDIO COMPARISON")
        print("="*70)
        
        source_dir = config.SOURCE_DIR
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
        
        if not os.path.exists(source_dir):
            print(f"⚠️  Source directory not found: {source_dir}")
            return
        
        if not os.path.exists(test_dir):
            print(f"⚠️  Test directory not found: {test_dir}")
            return
        
        print("TRAINING SOURCE AUDIO ANALYSIS:")
        # Analyze training source
        for class_name, folder_name in config.CLASS_MAPPING.items():
            class_path = os.path.join(source_dir, folder_name)
            if not os.path.exists(class_path):
                continue
            
            wav_files = list(Path(class_path).glob('*.wav'))[:5]
            
            if wav_files:
                energies = []
                durations = []
                for wav_file in wav_files:
                    try:
                        y, sr = sf.read(str(wav_file))
                        if len(y.shape) > 1:
                            y = np.mean(y, axis=1)
                        energies.append(np.sqrt(np.mean(y**2)))
                        durations.append(len(y) / sr)
                    except:
                        continue
                
                if energies:
                    print(f"\n  {class_name}:")
                    print(f"    Avg Energy: {np.mean(energies):.4f}")
                    print(f"    Avg Duration: {np.mean(durations):.2f}s")
                    print(f"    Energy Range: [{min(energies):.4f}, {max(energies):.4f}]")
        
        print("\n\nTEST AUDIO ANALYSIS:")
        test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3'))][:10]
        
        if test_files:
            energies = []
            durations = []
            for test_file in test_files:
                try:
                    y, sr = sf.read(os.path.join(test_dir, test_file))
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                    energies.append(np.sqrt(np.mean(y**2)))
                    durations.append(len(y) / sr)
                except:
                    continue
            
            if energies:
                print(f"  Avg Energy: {np.mean(energies):.4f}")
                print(f"  Avg Duration: {np.mean(durations):.2f}s")
                print(f"  Energy Range: [{min(energies):.4f}, {max(energies):.4f}]")
                
                # Compare
                print("\n  COMPARISON:")
                print(f"    If test energy is 10x lower than training: Files are too quiet")
                print(f"    If test duration is <1s: May not capture full cry/bark")
        
        print()
    
    def test_5_prediction_on_test_samples(self):
        """TEST 5: Actually predict on test samples and show detailed results"""
        print("="*70)
        print("TEST 5: DETAILED PREDICTION ON TEST SAMPLES")
        print("="*70)
        
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
        if not os.path.exists(test_dir):
            print("⚠️  No test directory found.")
            return
        
        test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3', '.ogg'))][:10]
        
        if not test_files:
            print("❌ No test files found!")
            return
        
        print(f"Testing {len(test_files)} files...\n")
        print(f"{'FILE':<35} | {'PRED':<12} | {'CONF':<6} | {'BABY':<6} | {'DOG':<6} | {'UNK':<6}")
        print("-"*70)
        
        for test_file in test_files:
            path = os.path.join(test_dir, test_file)
            
            try:
                # Extract features using ProPreprocessor
                feat = self.prep.extract_features(path, augment=False)
                
                # Predict
                feat_input = feat[np.newaxis, ..., np.newaxis]
                probs = self.model.predict(feat_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_label = config.CLASS_NAMES[pred_idx]
                conf = probs[pred_idx]
                
                display_name = (test_file[:32] + '...') if len(test_file) > 32 else test_file
                print(f"{display_name:<35} | {pred_label:<12} | {conf:.3f}  | {probs[0]:.3f}  | {probs[1]:.3f}  | {probs[2]:.3f}")
                
            except Exception as e:
                print(f"{test_file:<35} | ERROR: {str(e)[:40]}")
        
        print()
    
    def test_6_visualize_spectrograms(self):
        """TEST 6: Visualize spectrograms to see if they make sense"""
        print("="*70)
        print("TEST 6: SPECTROGRAM VISUALIZATION")
        print("="*70)
        
        # Get one sample from each training class
        training_samples = {}
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.PROCESSED_DIR, class_name)
            if os.path.exists(class_dir):
                npy_files = [f for f in os.listdir(class_dir) if f.endswith('_clean.npy')]
                if npy_files:
                    training_samples[class_name] = np.load(os.path.join(class_dir, npy_files[0]))
        
        # Get test samples
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
        test_samples = {}
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3'))][:3]
            for test_file in test_files:
                try:
                    feat = self.prep.extract_features(os.path.join(test_dir, test_file), augment=False)
                    test_samples[test_file] = feat
                except:
                    pass
        
        # Create visualization
        n_samples = len(training_samples) + len(test_samples)
        if n_samples == 0:
            print("No samples to visualize")
            return
        
        fig, axes = plt.subplots(2, max(3, (n_samples+1)//2), figsize=(18, 8))
        axes = axes.flatten()
        
        idx = 0
        for class_name, feat in training_samples.items():
            axes[idx].imshow(feat, aspect='auto', origin='lower', cmap='viridis')
            axes[idx].set_title(f'TRAIN: {class_name}')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Mel Bins')
            idx += 1
        
        for test_name, feat in test_samples.items():
            axes[idx].imshow(feat, aspect='auto', origin='lower', cmap='viridis')
            axes[idx].set_title(f'TEST: {test_name[:15]}...')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Mel Bins')
            idx += 1
        
        # Hide extra subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(config.LOG_DIR, 'diagnostic_spectrograms.png')
        plt.savefig(output_path)
        print(f"✅ Spectrograms saved to: {output_path}")
        print("   Look for:")
        print("   - Test spectrograms that are mostly dark/empty (silent audio)")
        print("   - Test spectrograms with very different patterns than training")
        print()
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("\n" + "🚨"*35)
        print("     EMERGENCY DIAGNOSTIC SUITE")
        print("🚨"*35 + "\n")
        
        if not self.model:
            print("❌ Cannot run tests - model failed to load")
            return
        
        self.test_1_model_on_training_samples()
        self.test_2_compare_train_vs_test_features()
        self.test_3_audio_quality_check()
        self.test_4_source_data_vs_test_data()
        self.test_5_prediction_on_test_samples()
        self.test_6_visualize_spectrograms()
        
        print("="*70)
        print("DIAGNOSTIC COMPLETE")
        print("="*70)
        print("\nNEXT STEPS:")
        print("1. Review the test results above")
        print("2. Check the diagnostic_spectrograms.png in logs/")
        print("3. Share the output with me for analysis")
        print()

if __name__ == '__main__':
    diag = EmergencyDiagnostic()
    diag.run_all_tests()