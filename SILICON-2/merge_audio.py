import os
from pydub import AudioSegment

input_dirs = ["DvC/Cats"]
output_dir = "DvC/Merged_Cats"
os.makedirs(output_dir, exist_ok=True)

wav_files = []
for dir_path in input_dirs:
    wav_files.extend([os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if f.endswith(".wav")])

output_count = 1
current_audio = AudioSegment.empty()
current_duration = 0

for wav_file in wav_files:
    audio = AudioSegment.from_wav(wav_file)
    duration = len(audio) / 1000.0
    
    current_audio += audio
    current_duration += duration
    
    if current_duration >= 7.0:
        output_path = os.path.join(output_dir, f"DvC_merged_{output_count:03d}.wav")
        current_audio.export(output_path, format="wav")
        print(f"Created: merged_{output_count:03d}.wav ({current_duration:.2f}s)")
        output_count += 1
        current_audio = AudioSegment.empty()
        current_duration = 0

if len(current_audio) > 0:
    output_path = os.path.join(output_dir, f"merged_{output_count:03d}.wav")
    current_audio.export(output_path, format="wav")
    print(f"Created: merged_{output_count:03d}.wav ({current_duration:.2f}s)")
