"""
SILICON-14: Audit - Listen to the Brain's Input
Saves the processed audio EXACTLY as the model hears it.
"""

import os
import random
import soundfile as sf
from tqdm import tqdm
import config
from preprocess import ProPreprocessor

def audit_audio():
    prep = ProPreprocessor()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_samples')
    os.makedirs(output_dir, exist_ok=True)
    
    source_dir = config.SOURCE_DIR
    
    print("\n" + "="*60)
    print("      SILICON-14: AUDIO PREPROCESSING AUDIT")
    print("      Saving samples to: audit_samples/")
    print("="*60)

    for class_name, folder_name in config.CLASS_MAPPING.items():
        class_dir = os.path.join(source_dir, folder_name)
        if not os.path.exists(class_dir):
            print(f"❌ Missing: {class_dir}")
            continue
            
        all_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        # Pick 20 random files
        sample_files = random.sample(all_files, min(20, len(all_files)))
        
        class_out = os.path.join(output_dir, class_name)
        os.makedirs(class_out, exist_ok=True)
        
        print(f"Processing {class_name} samples...")
        for f_name in tqdm(sample_files):
            input_path = os.path.join(class_dir, f_name)
            
            # Get the raw processed audio (with Notch, Augment, Windowing, Warm-up)
            # We enable augment=True so you can hear why the model is so "tough"
            processed_audio = prep.get_processed_audio(input_path, augment=True)
            
            output_path = os.path.join(class_out, f"audit_{f_name}")
            sf.write(output_path, processed_audio, config.SAMPLE_RATE_HZ)

    print("\n✅ Audit Complete! You can now listen to the files in 'audit_samples/' folder.")
    print("Listen for: ")
    print("1. The 500ms static 'hiss' at the start (Noise Warm-up)")
    print("2. The random gain/noise (Augmentation)")
    print("3. How the dog barks are perfectly centered (Energy Windowing)")
    print("="*60 + "\n")

if __name__ == '__main__':
    audit_audio()
