import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_aif_to_wav():
    src_dir = Path("_temp_dog/animal_sounds/dogs")
    dest_dir = Path("Dog")
    dest_dir.mkdir(exist_ok=True)
    
    aif_files = list(src_dir.glob("*.aif"))
    print(f"Found {len(aif_files)} AIF files to convert.")
    
    converted_count = 0
    for aif_path in tqdm(aif_files, desc="Converting AIF to WAV (lossless)"):
        wav_filename = f"asc_{aif_path.stem}.wav"
        wav_path = dest_dir / wav_filename
        
        # ffmpeg command to convert pcm_s16be (AIF) to pcm_s16le (WAV) losslessly
        # -c:a pcm_s16le ensures we use 16-bit PCM (same as source)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(aif_path),
            "-c:a", "pcm_s16le",
            "-loglevel", "error",
            str(wav_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            converted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {aif_path.name}: {e}")
            
    print(f"\nSuccessfully converted {converted_count} files to lossless WAV in the 'Dog' folder.")

if __name__ == "__main__":
    convert_aif_to_wav()
