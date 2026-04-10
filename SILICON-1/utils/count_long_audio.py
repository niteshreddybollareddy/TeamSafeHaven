import soundfile as sf
from pathlib import Path

audio_dir = Path("extracted_audio")
count = 0

for wav_file in audio_dir.rglob("*.wav"):
    info = sf.info(wav_file)
    if info.duration >= 9.0:
        count += 1

print(f"Total .wav files >= 9 seconds: {count}")
