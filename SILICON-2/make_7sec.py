from pydub import AudioSegment
from pydub.generators import WhiteNoise
import os
import random

input_dir = "raw_cat"
output_dir = "processed_7sec"
os.makedirs(output_dir, exist_ok=True)

wav_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")])

output_count = 1
current_audio = AudioSegment.empty()

def generate_home_noise(duration):
    noise = AudioSegment.empty()
    segments = random.randint(3, 6)
    seg_duration = duration // segments
    for _ in range(segments):
        gain = random.uniform(-50, -35)
        seg = WhiteNoise().to_audio_segment(duration=seg_duration).apply_gain(gain).low_pass_filter(2000)
        noise += seg
    return noise[:duration]

for wav_file in wav_files:
    audio = AudioSegment.from_wav(wav_file)
    if len(current_audio) > 0:
        gap_duration = random.randint(500, 1000)
        current_audio += generate_home_noise(gap_duration)
    current_audio += audio
    
    while len(current_audio) >= 7000:
        target_duration = min(len(current_audio), 8000)
        chunk = current_audio[:target_duration]
        
        output_path = os.path.join(output_dir, f"audio_{output_count:03d}.wav")
        chunk.export(output_path, format="wav")
        print(f"Created: audio_{output_count:03d}.wav ({len(chunk)/1000:.2f}s)")
        output_count += 1
        current_audio = current_audio[target_duration:]

if len(current_audio) > 0:
    output_path = os.path.join(output_dir, f"audio_{output_count:03d}.wav")
    current_audio.export(output_path, format="wav")
    print(f"Created: audio_{output_count:03d}.wav ({len(current_audio)/1000:.2f}s)")
