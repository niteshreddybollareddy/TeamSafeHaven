"""
SILICON-16: IN-DEPTH DATASET ANALYZER
Performs comprehensive analysis of audio files to help determine 
optimal model parameters (duration, windows, energy thresholds).
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# Configuration
BASE_DIR = "/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-16"
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
CLASSES = ['baby_cry', 'unknown']
OUTPUT_CSV = os.path.join(BASE_DIR, 'dataset_analysis_report.csv')

def analyze_file(file_path, label):
    try:
        # File-level info using soundfile (faster for metadata)
        info = sf.info(file_path)
        
        # Load audio for in-depth analysis
        # Using sr=None to get native sample rate
        y, sr = librosa.load(file_path, sr=None)
        
        if len(y) == 0:
            return {"file": file_path, "error": "Empty file"}

        # Time-domain stats
        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        max_amp = np.max(np.abs(y))
        min_amp = np.min(np.abs(y))
        mean_amp = np.mean(y)
        dc_offset = np.mean(y)
        
        # Length analysis
        total_samples = len(y)
        
        # Silence & Activity
        # Using a -30dB threshold for silence
        intervals = librosa.effects.split(y, top_db=30)
        active_duration = sum([(end - start) for start, end in intervals]) / sr
        silence_ratio = (duration - active_duration) / duration if duration > 0 else 0
        num_silent_segments = len(intervals) # This is actually non-silent blocks
        
        # Energy
        avg_energy = np.mean(y**2)
        energy_var = np.var(y**2)
        
        # Frequency-related (Lightweight)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        return {
            "ID": os.path.basename(file_path),
            "Class": label,
            "Path": file_path,
            "SR (Hz)": sr,
            "Channels": info.channels,
            "Format": info.format,
            "Duration (s)": round(duration, 3),
            "Samples": total_samples,
            "Max Amp": round(max_amp, 4),
            "RMS Energy": round(rms, 4),
            "DC Offset": round(dc_offset, 6),
            "Silence Ratio (%)": round(silence_ratio * 100, 2),
            "Active Duration (s)": round(active_duration, 3),
            "Avg Energy": round(avg_energy, 6),
            "Spec Centroid": round(spec_centroid, 2),
            "Spec Bandwidth": round(spec_bandwidth, 2),
            "Valid": True
        }
    except Exception as e:
        return {"ID": os.path.basename(file_path), "Class": label, "Valid": False, "Error": str(e)}

def run_analysis():
    print("\n" + "="*60)
    print("      SILICON-16: DATASET DEEP ANALYZER")
    print("="*60)
    
    results = []
    
    for label in CLASSES:
        class_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_dir):
            print(f"⚠ Folder not found: {class_dir}")
            continue
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.endswith(('.wav', '.mp3', '.ogg'))]
        print(f"🔍 Analyzing {label}: {len(files)} files...")
        
        for f in tqdm(files):
            res = analyze_file(f, label)
            results.append(res)
            
    df = pd.DataFrame(results)
    valid_df = df[df['Valid'] == True].copy()
    
    # Save full report
    df.to_csv(OUTPUT_CSV, index=False)
    
    # --- DATASET LEVEL SUMMARY ---
    print("\n" + "-"*60)
    print("               DATASET SUMMARY REPORT")
    print("-"*60)
    
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

    # --- VALIDATION CHECKS ---
    print("\n" + "-"*60)
    print("                  VALDIATION CHECKS")
    print("-"*60)
    
    # 1. Corrupted
    corrupted = df[df['Valid'] == False]
    print(f"❌ Corrupted/Unreadable Files: {len(corrupted)}")
    if len(corrupted) > 0:
        print(corrupted[['ID', 'Error']].head())
        
    # 2. SR Mismatch (Target is 16000)
    sr_mismatch = valid_df[valid_df['SR (Hz)'] != 16000]
    print(f"⚠ Non-16kHz Files (Will need resampling): {len(sr_mismatch)}")
    
    # 3. Stereo Check
    stereo = valid_df[valid_df['Channels'] > 1]
    print(f"⚠ Stereo Files (Will need mono-downmix): {len(stereo)}")
    
    # 4. Outliers (Short Duration)
    too_short = valid_df[valid_df['Duration (s)'] < 1.0]
    print(f"‼ Outliers - Too Short (< 1.0s): {len(too_short)}")
    
    # 5. Outliers (Very Quiet)
    too_quiet = valid_df[valid_df['RMS Energy'] < 0.001]
    print(f"‼ Outliers - Too Quiet (RMS < 0.001): {len(too_quiet)}")

    print(f"\n✅ Full analysis saved to: {OUTPUT_CSV}")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_analysis()
