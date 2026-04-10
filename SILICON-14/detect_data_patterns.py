import os
import librosa
import numpy as np
import json
from tqdm import tqdm

def analyze_audio_directory(dir_path, class_name):
    if not os.path.exists(dir_path):
        return None
    
    files = [f for f in os.listdir(dir_path) if f.endswith(('.wav', '.ogg', '.mp3'))]
    if not files:
        return None
        
    stats = {
        'count': len(files),
        'durations': [],
        'rms_max': [],
        'rms_mean': [],
        'silence_percent': []
    }
    
    # Analyze a representative sample (up to 50 files per class for speed)
    sample_files = files[:50]
    
    for f in sample_files:
        try:
            path = os.path.join(dir_path, f)
            y, sr = librosa.load(path, sr=16000)
            
            # Duration
            stats['durations'].append(len(y)/sr)
            
            # Energy
            rms = librosa.feature.rms(y=y).squeeze()
            stats['rms_max'].append(float(np.max(rms)))
            stats['rms_mean'].append(float(np.mean(rms)))
            
            # Silence (Threshold: -40dB)
            non_silent_intervals = librosa.effects.split(y, top_db=40)
            active_len = sum([i[1] - i[0] for i in non_silent_intervals])
            stats['silence_percent'].append(1.0 - (active_len / len(y)))
            
        except Exception as e:
            continue
            
    return {
        'class': class_name,
        'path': dir_path,
        'total_files': stats['count'],
        'avg_duration': np.mean(stats['durations']) if stats['durations'] else 0,
        'max_duration': np.max(stats['durations']) if stats['durations'] else 0,
        'avg_rms': np.mean(stats['rms_mean']) if stats['rms_mean'] else 0,
        'avg_silence': np.mean(stats['silence_percent']) if stats['silence_percent'] else 0
    }

def main():
    paths_to_check = {
        'SILICON-7': {
            'Baby Cry': '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-7/baby cry',
            'Dog': '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-7/dog',
            'Unknown': '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-7/unknown'
        },
        'SILICON-10': {
            'Baby Cry': '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-10/Dataset/baby cry',
            'Dog': '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-10/Dataset/dog',
            'Unknown': '/Users/theepireddykarthikreddy/Desktop/SILICON/SILICON-10/Dataset/unknown'
        }
    }
    
    report = []
    
    print("🚀 Starting Audio Pattern Analysis...")
    for source, classes in paths_to_check.items():
        print(f"Checking {source}...")
        for cls_name, cls_path in classes.items():
            res = analyze_audio_directory(cls_path, cls_name)
            if res:
                res['source'] = source
                report.append(res)

    with open('audio_analysis_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("      SILICON AUDIO PATTERN ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        for r in report:
            f.write(f"SOURCE: {r['source']} | CLASS: {r['class']}\n")
            f.write(f"  Path:          {r['path']}\n")
            f.write(f"  Total Files:   {r['total_files']}\n")
            f.write(f"  Avg Duration:  {r['avg_duration']:.2f}s\n")
            f.write(f"  Max Duration:  {r['max_duration']:.2f}s\n")
            f.write(f"  Avg Loudness:  {r['avg_rms']:.4f}\n")
            f.write(f"  Avg Silence:   {r['avg_silence']:.2%}\n")
            f.write("-" * 30 + "\n")
            
    print("✅ Analysis Complete! Check audio_analysis_report.txt")

if __name__ == '__main__':
    main()
