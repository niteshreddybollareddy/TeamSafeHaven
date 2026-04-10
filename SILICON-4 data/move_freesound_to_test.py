import os
import shutil
import glob

source_dir = "FINAL_CLEANED_SOUNDS/Object Drop"
dest_dir = "test"
os.makedirs(dest_dir, exist_ok=True)

files = glob.glob(os.path.join(source_dir, "freesound_*.wav"))

for f in files:
    shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))
    
print(f"Moved {len(files)} files starting with freesound_ to {dest_dir}")
