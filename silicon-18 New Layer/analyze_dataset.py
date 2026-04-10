"""
Analyzes audio files in the dataset to figure out good model parameters
like duration, window sizes, energy thresholds, etc.
"""

import os
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
CLASSES = ['baby_cry', 'dog', 'unknown']
OUTPUT_CSV = os.path.join(BASE_DIR, 'dataset_analysis_report.csv')

def analyze_file(file_path, label):
    try:
        # File-level info using soundfile
        info = sf.info(file_path)
        y, sr = sf.read(file_path, always_2d=False)
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        
        if len(y) == 0:
            return {"file": file_path, "error": "Empty file"}

        # Time-domain stats
        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        max_amp = np.max(np.abs(y))
        dc_offset = np.mean(y)
        
        # Manual VAD (100ms frames)
        frame_len = int(sr * 0.1)
        hop = frame_len // 2
        active_samples = 0
        if len(y) > frame_len:
            for i in range(0, len(y) - frame_len, hop):
                if np.sqrt(np.mean(y[i:i+frame_len]**2)) > 0.005:
                    active_samples += hop
        
        active_duration = active_samples / sr
        silence_ratio = (duration - active_duration) / duration if duration > 0 else 0
        
        # Manual Spectral Features (Full signal FFT)
        if len(y) > 512:
            n_fft = min(len(y), 2048)
            spec = np.abs(np.fft.rfft(y[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1/sr)
            
            # Centroid = sum(f * p) / sum(p)
            spec_centroid = np.sum(freqs * spec) / np.sum(spec)
            # Bandwidth = (sum(p * (f - centroid)^2)) ^ 0.5
            spec_bandwidth = np.sqrt(np.sum(spec * (freqs - spec_centroid)**2) / np.sum(spec))
        else:
            spec_centroid, spec_bandwidth = 0, 0
        
        return {
            "ID": os.path.basename(file_path),
            "Class": label,
            "Path": file_path,
            "SR (Hz)": sr,
            "Channels": info.channels,
            "Format": info.format,
            "Duration (s)": round(duration, 3),
            "Samples": len(y),
            "Max Amp": round(max_amp, 4),
            "RMS Energy": round(rms, 4),
            "DC Offset": round(dc_offset, 6),
            "Silence Ratio (%)": round(silence_ratio * 100, 2),
            "Active Duration (s)": round(active_duration, 3),
            "Avg Energy": round(np.mean(y**2), 6),
            "Spec Centroid": round(spec_centroid, 2),
            "Spec Bandwidth": round(spec_bandwidth, 2),
            "Valid": True
        }
    except Exception as e:
        return {"ID": os.path.basename(file_path), "Class": label, "Valid": False, "Error": str(e)}

def run_analysis():
    print("\n" + "="*60)
    print("      SILICON-18: DATASET DEEP ANALYZER")
    print("="*60)
    
    results = []
    
    for label in CLASSES:
        class_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_dir):
            print(f"Warning: folder not found: {class_dir}")
            continue
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.endswith(('.wav', '.mp3', '.ogg'))]
        print(f"Analyzing {label}: {len(files)} files...")
        
        for f in tqdm(files):
            res = analyze_file(f, label)
            results.append(res)
            
    if not results:
        print("No files found to analyze!")
        return

    df = pd.DataFrame(results)
    valid_df = df[df['Valid'] == True].copy()
    
    # Save full report
    df.to_csv(OUTPUT_CSV, index=False)
    
    # print summary per class
    print("\n--- Dataset Summary ---")
    
    for label in CLASSES:
        pdf = valid_df[valid_df['Class'] == label]
        if pdf.empty: continue
        
        print(f"\n[ CLASS: {label.upper()} ]")
        print(f"  Count: {len(pdf)} files")
        print(f"  Duration: Mean={pdf['Duration (s)'].mean():.2f}s | Min={pdf['Duration (s)'].min():.2f}s | Max={pdf['Duration (s)'].max():.2f}s")
        print(f"  Median Duration: {pdf['Duration (s)'].median():.2f}s")
        print(f"  Sample Rates: {pdf['SR (Hz)'].unique()}")
        print(f"  RMS Loudness: Mean={pdf['RMS Energy'].mean():.4f}")
        print(f"  Avg Silence: {pdf['Silence Ratio (%)'].mean():.1f}%")
        print(f"  Peak Freq (Centroid): {pdf['Spec Centroid'].mean():.2f} Hz")

    # check for any issues in the data
    print("\n--- Validation Checks ---")
    
    # corrupted files
    corrupted = df[df['Valid'] == False]
    print(f"Corrupted/unreadable files: {len(corrupted)}")
    if len(corrupted) > 0:
        print(corrupted[['ID', 'Error']].head())
        
    # sample rate check (we want 16kHz)
    sr_mismatch = valid_df[valid_df['SR (Hz)'] != 16000]
    print(f"Non-16kHz files (need resampling): {len(sr_mismatch)}")
    
    # stereo check
    stereo = valid_df[valid_df['Channels'] > 1]
    print(f"Stereo files (need mono downmix): {len(stereo)}")
    
    # short clips
    too_short = valid_df[valid_df['Duration (s)'] < 1.0]
    print(f"Too short (< 1.0s): {len(too_short)}")
    
    # 5. Outliers (Very Quiet)
    too_quiet = valid_df[valid_df['RMS Energy'] < 0.001]
    print(f"‼ Outliers - Too Quiet (RMS < 0.001): {len(too_quiet)}")

    print(f"\nFull analysis saved to: {OUTPUT_CSV}")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_analysis()
