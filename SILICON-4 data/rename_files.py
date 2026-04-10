import os

dir_path = "FINAL_CLEANED_SOUNDS/door_window"

wav_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".wav")])

for i, filename in enumerate(wav_files, 1):
    old_path = os.path.join(dir_path, filename)
    new_path = os.path.join(dir_path, f"Door_Window_audio_{i}.wav")
    os.rename(old_path, new_path)
