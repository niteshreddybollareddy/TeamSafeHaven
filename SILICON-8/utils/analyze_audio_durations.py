"""
Analyze audio file durations in the dataset
"""
import os
import soundfile as sf
from pathlib import Path
import numpy as np

DATASET_DIR = 'Dataset'
CLASS_NAMES = ['Baby Cry', 'Dog', 'Unknown', 'Unusual']

def analyze_durations():
    """Analyze duration of all audio files"""

    print("="*80)
    print("AUDIO DURATION ANALYSIS")
    print("="*80)

    all_durations = {}
    all_stats = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"\nWARNING: {class_dir} not found!")
            continue

        print(f"\n{'='*80}")
        print(f"CLASS: {class_name}")
        print(f"{'='*80}")

        audio_files = list(Path(class_dir).glob('*.wav'))
        audio_files.extend(list(Path(class_dir).glob('*.mp3')))
        audio_files.extend(list(Path(class_dir).glob('*.flac')))
        audio_files.extend(list(Path(class_dir).glob('*.ogg')))

        durations = []

        for audio_file in audio_files:
            try:
                info = sf.info(str(audio_file))
                duration = info.duration
                durations.append(duration)
            except Exception as e:
                print(f"Error reading {audio_file.name}: {e}")

        if not durations:
            print(f"No audio files found in {class_name}")
            continue

        durations = np.array(durations)
        all_durations[class_name] = durations

        # Statistics
        print(f"Total files: {len(durations)}")
        print(f"\nDuration Statistics:")
        print(f"  Min:    {durations.min():.2f}s")
        print(f"  Max:    {durations.max():.2f}s")
        print(f"  Mean:   {durations.mean():.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")
        print(f"  Std:    {durations.std():.2f}s")

        # Distribution
        print(f"\nDuration Distribution:")
        bins = [0, 1, 2, 3, 5, 8, 10, 15, 20, 100]
        bin_labels = ['0-1s', '1-2s', '2-3s', '3-5s', '5-8s', '8-10s', '10-15s', '15-20s', '>20s']

        for i in range(len(bins)-1):
            count = np.sum((durations >= bins[i]) & (durations < bins[i+1]))
            percentage = (count / len(durations)) * 100
            print(f"  {bin_labels[i]:>8}: {count:4d} files ({percentage:5.1f}%)")

        # Long files (>10s, >15s, >20s)
        long_10s = np.sum(durations > 10)
        long_15s = np.sum(durations > 15)
        long_20s = np.sum(durations > 20)

        print(f"\nLong Files:")
        print(f"  > 10s: {long_10s:4d} files ({long_10s/len(durations)*100:5.1f}%)")
        print(f"  > 15s: {long_15s:4d} files ({long_15s/len(durations)*100:5.1f}%)")
        print(f"  > 20s: {long_20s:4d} files ({long_20s/len(durations)*100:5.1f}%)")

        all_stats.append({
            'class': class_name,
            'total': len(durations),
            'min': durations.min(),
            'max': durations.max(),
            'mean': durations.mean(),
            'median': np.median(durations),
            'long_10s': long_10s,
            'long_15s': long_15s,
            'long_20s': long_20s
        })

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    total_files = sum([s['total'] for s in all_stats])
    total_long_10s = sum([s['long_10s'] for s in all_stats])
    total_long_15s = sum([s['long_15s'] for s in all_stats])
    total_long_20s = sum([s['long_20s'] for s in all_stats])

    print(f"Total files: {total_files}")
    print(f"\nLong Files (causing high memory/computation):")
    print(f"  > 10s: {total_long_10s:4d} files ({total_long_10s/total_files*100:5.1f}%)")
    print(f"  > 15s: {total_long_15s:4d} files ({total_long_15s/total_files*100:5.1f}%)")
    print(f"  > 20s: {total_long_20s:4d} files ({total_long_20s/total_files*100:5.1f}%)")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if total_long_10s / total_files < 0.1:  # Less than 10%
        print("✓ Only a small fraction of files are >10s")
        print("  → Option: Trim all audio to 10s max (minimal data loss)")

    if total_long_15s / total_files < 0.05:  # Less than 5%
        print("✓ Very few files are >15s")
        print("  → Option: Remove or trim files >15s")

    if total_long_20s > 0:
        print(f"⚠ {total_long_20s} files are >20s (extreme length)")
        print("  → Recommendation: Trim to 8-10s max or remove outliers")

    # Frame count estimates
    print(f"\n{'='*80}")
    print("FRAME COUNT ESTIMATES (for different hop lengths)")
    print(f"{'='*80}")

    for class_name in CLASS_NAMES:
        if class_name not in all_durations:
            continue

        durations = all_durations[class_name]
        max_duration = durations.max()

        # Calculate frame counts for different hop lengths
        hop_160_frames = int(max_duration * (16000 / 160))  # hop=160 (MLTK)
        hop_320_frames = int(max_duration * (16000 / 320))  # hop=320 (SILICON-7)

        print(f"\n{class_name} (max: {max_duration:.1f}s):")
        print(f"  With hop=160 (MLTK):     ~{hop_160_frames:4d} frames")
        print(f"  With hop=320 (SILICON-7): ~{hop_320_frames:4d} frames")

        if hop_160_frames > 1500:
            print(f"  ⚠ WARNING: Very long sequence with hop=160!")
        if hop_320_frames > 800:
            print(f"  ⚠ WARNING: Long sequence with hop=320!")

if __name__ == '__main__':
    analyze_durations()
