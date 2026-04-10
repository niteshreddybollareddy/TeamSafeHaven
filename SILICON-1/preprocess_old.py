import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import config

def load_and_preprocess_audio(audio_path, sr=config.SAMPLE_RATE):
    """Load audio with preprocessing"""
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Remove silence from beginning and end
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize audio to [-1, 1] range
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max()
    
    return y

def extract_melspectrogram(audio_path, sr=config.SAMPLE_RATE, n_mels=config.N_MELS):
    """Extract mel-spectrogram from audio file with improved settings"""
    y = load_and_preprocess_audio(audio_path, sr)
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=config.N_FFT, 
        hop_length=config.HOP_LENGTH,
        fmin=50,  # Focus on relevant frequency range for baby cries
        fmax=8000,  # Most baby cry energy is below 8kHz
        power=2.0  # Use power spectrogram (squared magnitude)
    )
    
    # Convert to dB scale with better reference
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    
    return mel_spec_db, len(y) / sr

def pad_or_truncate(mel_spec, max_frames=config.MAX_TIME_FRAMES):
    """Pad or truncate to fixed length with edge padding"""
    current_frames = mel_spec.shape[1]
    
    if current_frames < max_frames:
        # Use edge padding instead of zero padding
        pad_width = max_frames - current_frames
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='edge')
    else:
        # Take center crop if too long
        start = (current_frames - max_frames) // 2
        mel_spec = mel_spec[:, start:start + max_frames]
    
    return mel_spec

def save_melspectrogram(mel_spec, output_path):
    """Save mel-spectrogram as numpy array"""
    np.save(output_path, mel_spec)

def visualize_sample(mel_spec, title, output_path):
    """Visualize a mel-spectrogram sample"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=config.SAMPLE_RATE, 
                            hop_length=config.HOP_LENGTH,
                            x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def analyze_dataset_stats():
    """Analyze dataset statistics before processing"""
    print("="*60)
    print("Dataset Analysis")
    print("="*60)
    
    for class_name in config.CLASS_NAMES:
        input_dir = os.path.join(config.DATASET_DIR, class_name)
        audio_files = list(Path(input_dir).glob('*.wav')) + list(Path(input_dir).glob('*.ogg'))
        
        if not audio_files:
            print(f"\nWARNING: No audio files found in {input_dir}")
            continue
        
        durations = []
        sample_rates = []
        
        print(f"\n{class_name.upper()} class:")
        print(f"  Total files: {len(audio_files)}")
        
        # Sample a few files for analysis
        for audio_file in audio_files[:min(10, len(audio_files))]:
            try:
                y, sr = librosa.load(str(audio_file), sr=None)
                duration = len(y) / sr
                durations.append(duration)
                sample_rates.append(sr)
            except Exception as e:
                print(f"  Error reading {audio_file.name}: {e}")
        
        if durations:
            print(f"  Sample durations (seconds):")
            print(f"    Min: {min(durations):.2f}, Max: {max(durations):.2f}, Mean: {np.mean(durations):.2f}")
            print(f"  Original sample rates: {set(sample_rates)}")
            
            # Calculate expected frames
            expected_frames = np.array(durations) * config.SAMPLE_RATE / config.HOP_LENGTH
            print(f"  Expected mel-spec frames: {expected_frames.mean():.0f} (target: {config.MAX_TIME_FRAMES})")

def process_dataset():
    """Process all audio files and save mel-spectrograms"""
    
    # First analyze the dataset
    analyze_dataset_stats()
    
    print("\n" + "="*60)
    print("Starting Processing")
    print("="*60)
    
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    # Create visualization directory
    viz_dir = os.path.join(config.PROCESSED_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    all_stats = {'total': 0, 'errors': 0, 'warnings': 0}
    
    for class_name in config.CLASS_NAMES:
        input_dir = os.path.join(config.DATASET_DIR, class_name)
        output_dir = os.path.join(config.PROCESSED_DIR, class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = list(Path(input_dir).glob('*.wav')) + list(Path(input_dir).glob('*.ogg'))
        print(f'\nProcessing {class_name}: {len(audio_files)} files')
        
        class_stats = {'processed': 0, 'errors': 0, 'short': 0, 'long': 0}
        
        for i, audio_file in enumerate(tqdm(audio_files, desc=f'{class_name}', ncols=100)):
            try:
                mel_spec, duration = extract_melspectrogram(str(audio_file))
                
                # Check duration warnings
                if duration < 3.0:
                    class_stats['short'] += 1
                elif duration > 10.0:
                    class_stats['long'] += 1
                
                mel_spec = pad_or_truncate(mel_spec)
                
                output_path = os.path.join(output_dir, audio_file.stem + '.npy')
                save_melspectrogram(mel_spec, output_path)
                
                # Save visualization for first 3 samples of each class
                if i < 3:
                    viz_path = os.path.join(viz_dir, f'{class_name}_{i}.png')
                    visualize_sample(mel_spec, f'{class_name} - {audio_file.name}', viz_path)
                
                class_stats['processed'] += 1
                all_stats['total'] += 1
                
            except Exception as e:
                print(f'\nError processing {audio_file}: {e}')
                class_stats['errors'] += 1
                all_stats['errors'] += 1
        
        # Print class statistics
        print(f'\n{class_name} Statistics:')
        print(f'  Successfully processed: {class_stats["processed"]}')
        print(f'  Errors: {class_stats["errors"]}')
        print(f'  Short files (<3s): {class_stats["short"]}')
        print(f'  Long files (>10s): {class_stats["long"]}')
    
    print(f'\n{"="*60}')
    print('Processing Complete!')
    print(f'{"="*60}')
    print(f'Total files processed: {all_stats["total"]}')
    print(f'Total errors: {all_stats["errors"]}')
    print(f'Saved to: {config.PROCESSED_DIR}')
    print(f'Visualizations saved to: {viz_dir}')
    print(f'\nNext step: Run train.py to train the model')

if __name__ == '__main__':
    process_dataset()