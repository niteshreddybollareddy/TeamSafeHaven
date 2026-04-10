from pydub import AudioSegment
import os
import shutil

input_dir = "cat"
output_dir = "cat_temp"
os.makedirs(output_dir, exist_ok=True)

wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

for filename in wav_files:
    filepath = os.path.join(input_dir, filename)
    audio = AudioSegment.from_wav(filepath)
    duration = len(audio) / 1000
    
    if duration < 7:
        shutil.move(filepath, os.path.join(output_dir, filename))
        print(f"Moved: {filename} ({duration:.2f}s)")
