from pydub import AudioSegment
import os

input_dir = "dog random"

for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(input_dir, filename)
        wav_path = os.path.join(input_dir, filename.replace(".mp3", ".wav"))
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav", parameters=['-acodec', 'pcm_s16le'])
        print(f"Converted: {filename} -> {filename.replace('.mp3', '.wav')}")
