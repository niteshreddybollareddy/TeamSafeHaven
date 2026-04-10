"""
Copy audio files >10s to a separate directory for manual review
"""
import os
import shutil
import soundfile as sf
from pathlib import Path

DATASET_DIR = 'Dataset'
OUTPUT_DIR = 'long_files_review'  # Directory to copy long files
CLASS_NAMES = ['Baby Cry', 'Dog', 'Unknown', 'Unusual']
DURATION_THRESHOLD = 10.0  # seconds

def copy_long_files():
    """Copy all audio files longer than threshold to review directory"""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print(f"COPYING FILES > {DURATION_THRESHOLD}s FOR MANUAL REVIEW")
    print("="*80)

    total_copied = 0

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"\nWARNING: {class_dir} not found!")
            continue

        print(f"\n{'='*80}")
        print(f"CLASS: {class_name}")
        print(f"{'='*80}")

        # Create class subdirectory in output
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # Get all audio files
        audio_files = list(Path(class_dir).glob('*.wav'))
        audio_files.extend(list(Path(class_dir).glob('*.mp3')))
        audio_files.extend(list(Path(class_dir).glob('*.flac')))
        audio_files.extend(list(Path(class_dir).glob('*.ogg')))

        long_files = []

        for audio_file in audio_files:
            try:
                info = sf.info(str(audio_file))
                duration = info.duration

                if duration > DURATION_THRESHOLD:
                    long_files.append({
                        'path': audio_file,
                        'name': audio_file.name,
                        'duration': duration
                    })
            except Exception as e:
                print(f"Error reading {audio_file.name}: {e}")

        # Sort by duration (longest first)
        long_files.sort(key=lambda x: x['duration'], reverse=True)

        print(f"Found {len(long_files)} files > {DURATION_THRESHOLD}s")

        if long_files:
            print(f"\nCopying files to: {output_class_dir}/")
            print(f"{'File Name':<50} {'Duration':>10}")
            print("-" * 80)

            for file_info in long_files:
                src_path = file_info['path']
                dst_path = os.path.join(output_class_dir, file_info['name'])

                # Copy file
                shutil.copy2(src_path, dst_path)

                print(f"{file_info['name']:<50} {file_info['duration']:>9.2f}s")
                total_copied += 1
        else:
            print("No long files found in this class.")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total files copied: {total_copied}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"\nYou can now manually review these files and decide:")
    print(f"  1. Which files to keep (trim to 10s)")
    print(f"  2. Which files to remove (outliers/corrupt)")
    print(f"  3. Check if the important content is in the first 10s")

    # Create a summary file
    summary_file = os.path.join(OUTPUT_DIR, 'SUMMARY.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"LONG FILES (>{DURATION_THRESHOLD}s) SUMMARY\n")
        f.write("="*80 + "\n\n")

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(DATASET_DIR, class_name)
            if not os.path.exists(class_dir):
                continue

            f.write(f"\n{'='*80}\n")
            f.write(f"CLASS: {class_name}\n")
            f.write(f"{'='*80}\n\n")

            audio_files = list(Path(class_dir).glob('*.wav'))
            audio_files.extend(list(Path(class_dir).glob('*.mp3')))
            audio_files.extend(list(Path(class_dir).glob('*.flac')))
            audio_files.extend(list(Path(class_dir).glob('*.ogg')))

            long_files = []

            for audio_file in audio_files:
                try:
                    info = sf.info(str(audio_file))
                    duration = info.duration

                    if duration > DURATION_THRESHOLD:
                        long_files.append({
                            'name': audio_file.name,
                            'duration': duration
                        })
                except:
                    pass

            long_files.sort(key=lambda x: x['duration'], reverse=True)

            if long_files:
                f.write(f"Total: {len(long_files)} files\n\n")
                f.write(f"{'File Name':<50} {'Duration':>10}\n")
                f.write("-" * 80 + "\n")

                for file_info in long_files:
                    f.write(f"{file_info['name']:<50} {file_info['duration']:>9.2f}s\n")
            else:
                f.write("No long files found.\n")

    print(f"\nSummary saved to: {summary_file}")

if __name__ == '__main__':
    copy_long_files()
