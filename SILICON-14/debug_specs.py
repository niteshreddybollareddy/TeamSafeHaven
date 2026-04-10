import os
import numpy as np
import matplotlib.pyplot as plt
import config
from preprocess import ProPreprocessor

def debug_spectrogram(audio_path, output_name):
    prep = ProPreprocessor()
    # Extract features (augment=False)
    feat = prep.extract_features(audio_path, augment=False)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(feat, aspect='auto', origin='lower')
    plt.title(f"Spectrogram: {os.path.basename(audio_path)}")
    plt.colorbar(label='dB')
    plt.xlabel('Time (frames)')
    plt.ylabel('Mel bins')
    plt.savefig(output_name)
    print(f"Saved spectrogram debug to {output_name}")
    print(f"Feature shape: {feat.shape}, Stats: min={feat.min():.2f}, max={feat.max():.2f}, mean={feat.mean():.2f}")

if __name__ == '__main__':
    test_dir = '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-14/test'
    # Check a baby cry file
    baby_file = os.path.join(test_dir, 'baby-cry-101477.mp3')
    if os.path.exists(baby_file):
        debug_spectrogram(baby_file, 'debug_baby_test.png')
    
    # Check a training file (find one)
    train_dir = '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-10/Dataset/baby cry'
    if os.path.exists(train_dir):
        files = [f for f in os.listdir(train_dir) if f.endswith('.wav')]
        if files:
            debug_spectrogram(os.path.join(train_dir, files[0]), 'debug_baby_train.png')
