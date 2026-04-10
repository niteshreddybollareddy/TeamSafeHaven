import pandas as pd
import soundfile as sf
import os
from pathlib import Path

# Configuration
PARQUET_FILE = "0000.parquet"  # Change this to your parquet file path
OUTPUT_DIR = "extracted_audio_validation"  # Directory where audio files will be saved

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the parquet file
print("Loading parquet file...")
df = pd.read_parquet(PARQUET_FILE)

print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# First, inspect the structure of the audio data
print("\nInspecting audio data structure...")
first_audio = df.iloc[0]['audio']
print(f"Type: {type(first_audio)}")
print(f"Content: {first_audio if not isinstance(first_audio, (bytes, bytearray)) else 'Binary data'}")
if isinstance(first_audio, dict):
    print(f"Keys: {first_audio.keys()}")
print("\n")

# Extract audio files
for idx, row in df.iterrows():
    # Get audio data
    audio_data = row['audio']
    
    # Check if audio_data is a dictionary or needs to be decoded
    if isinstance(audio_data, dict):
        # If it's a dict, try to get the array and sampling_rate
        if 'array' in audio_data:
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
        elif 'bytes' in audio_data:
            # Audio might be stored as bytes, need to decode
            import io
            audio_bytes = audio_data['bytes']
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        else:
            print(f"Unknown audio format at index {idx}: {audio_data.keys()}")
            continue
    else:
        print(f"Audio data is not a dict at index {idx}, type: {type(audio_data)}")
        continue
    
    # Get metadata
    speaker_id = row['speaker_id']
    chapter_id = row['chapter_id']
    sample_id = row['id']
    
    # Create speaker directory
    speaker_dir = os.path.join(OUTPUT_DIR, f"speaker_{speaker_id}")
    os.makedirs(speaker_dir, exist_ok=True)
    
    # Create output filename
    output_file = os.path.join(speaker_dir, f"{sample_id}.wav")
    
    # Save audio file
    sf.write(output_file, audio_array, sampling_rate)
    
    # Progress indicator
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(df)} files...")

print(f"\nExtraction complete! Audio files saved to: {OUTPUT_DIR}")
print(f"Total files extracted: {len(df)}")