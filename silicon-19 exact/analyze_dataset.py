"""
SILICON-19: DATASET DEEP ANALYZER
"""

import os
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
CLASSES = ['baby_cry', 'dog', 'unknown']
OUTPUT_CSV = os.path.join(BASE_DIR, 'dataset_analysis_report.csv')

def analyze_file(file_path, label):
    try:
        info = sf.info(file_path)
        y, sr = sf.read(file_path, always_2d=False)
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        
        if len(y) == 0:
            return {"file": file_path, "error": "Empty file"}

        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        max_amp = np.max(np.abs(y))
        
        frame_len = int(sr * 0.1)
        hop = frame_len // 2
        active_samples = 0
        if len(y) > frame_len:
            for i in range(0, len(y) - frame_len, hop):
                if np.sqrt(np.mean(y[i:i+frame_len]**2)) > 0.005:
                    active_samples += hop
        
        active_duration = active_samples / sr
        silence_ratio = (duration - active_duration) / duration if duration > 0 else 0
        
        return {
            "ID": os.path.basename(file_path),
            "Class": label,
            "Path": file_path,
            "SR (Hz)": sr,
            "Channels": info.channels,
            "Duration (s)": round(duration, 3),
            "Max Amp": round(max_amp, 4),
            "RMS Energy": round(rms, 4),
            "Silence Ratio (%)": round(silence_ratio * 100, 2),
            "Spec Centroid": 0,
            "Valid": True
        }
    except Exception as e:
        return {"ID": os.path.basename(file_path), "Class": label, "Valid": False, "Error": str(e)}

def run_analysis():
    print("\n" + "="*60)
    print("      SILICON-19: DATASET ANALYZER")
    print("="*60)
    
    results = []
    for label in CLASSES:
        class_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_dir): continue
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
        print(f"Analyzing {label}: {len(files)} files...")
        for f in tqdm(files):
            results.append(analyze_file(f, label))
            
    if not results: return

    df = pd.DataFrame(results)
    valid_df = df[df['Valid'] == True].copy()
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n--- Summary ---")
    for label in CLASSES:
        pdf = valid_df[valid_df['Class'] == label]
        if pdf.empty: continue
        print(f"\n[ {label.upper()} ]")
        print(f"  Count: {len(pdf)}")
        print(f"  Avg Duration: {pdf['Duration (s)'].mean():.2f}s")
        print(f"  Avg Silence: {pdf['Silence Ratio (%)'].mean():.1f}%")

    print(f"\nReport: {OUTPUT_CSV}")

if __name__ == '__main__':
    run_analysis()
