import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def convert_wav_to_melspectrogram(wav_path, output_dir, sr=16000, n_mels=128):
    """Convert a single WAV file to mel-spectrogram image"""
    try:
        # Load audio file
        y, sr = librosa.load(wav_path, sr=sr)
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create output filename
        wav_name = Path(wav_path).stem
        output_path = os.path.join(output_dir, f"{wav_name}.png")
        
        # Plot and save
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-spectrogram: {wav_name}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return False

def process_all_wav_files(base_dir, output_base_dir):
    """Process all WAV files in the directory structure"""
    base_path = Path(base_dir)
    output_base_path = Path(output_base_dir)
    
    # Find all .wav files recursively
    wav_files = list(base_path.rglob("*.wav")) + list(base_path.rglob("*.WAV"))
    
    print(f"Found {len(wav_files)} WAV files")
    
    processed = 0
    for wav_file in wav_files:
        # Create corresponding output directory structure
        relative_path = wav_file.relative_to(base_path)
        output_dir = output_base_path / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if convert_wav_to_melspectrogram(str(wav_file), str(output_dir)):
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(wav_files)} files")
    
    print(f"Successfully processed {processed}/{len(wav_files)} files")

if __name__ == "__main__":
    # Set paths
    input_dir = "/Users/theepireddykarthikreddy/Desktop/SILICON"
    output_dir = "/Users/theepireddykarthikreddy/Desktop/SILICON/mel_spectrograms"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    process_all_wav_files(input_dir, output_dir)