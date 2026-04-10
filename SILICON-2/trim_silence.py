from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os

input_dir = "dog random"

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        audio = AudioSegment.from_wav(filepath)
        nonsilent_chunks = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        
        if nonsilent_chunks:
            trimmed = AudioSegment.empty()
            for start, end in nonsilent_chunks:
                trimmed += audio[start:end]
            trimmed.export(filepath, format="wav")
            print(f"Trimmed: {filename}")
