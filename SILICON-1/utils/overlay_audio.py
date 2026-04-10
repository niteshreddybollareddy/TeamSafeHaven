import librosa
import soundfile as sf
import numpy as np
import argparse

def overlay_audio(cry_file, voice_file, output_file, cry_volume=1.0, voice_volume=1.0):
    """Overlay baby cry and human voice audio files."""
    # Load audio files
    cry_audio, sr1 = librosa.load(cry_file, sr=None)
    voice_audio, sr2 = librosa.load(voice_file, sr=None)
    
    # Resample to same sample rate if different
    if sr1 != sr2:
        voice_audio = librosa.resample(voice_audio, orig_sr=sr2, target_sr=sr1)
        sr = sr1
    else:
        sr = sr1
    
    # Match lengths by padding shorter audio with zeros
    max_len = max(len(cry_audio), len(voice_audio))
    cry_audio = np.pad(cry_audio, (0, max_len - len(cry_audio)))
    voice_audio = np.pad(voice_audio, (0, max_len - len(voice_audio)))
    
    # Apply volume scaling and overlay
    overlayed = (cry_audio * cry_volume) + (voice_audio * voice_volume)
    
    # Normalize to prevent clipping
    overlayed = overlayed / np.max(np.abs(overlayed))
    
    # Save result
    sf.write(output_file, overlayed, sr)
    print(f"Overlayed audio saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay baby cry and human voice audio")
    parser.add_argument("cry_file", help="Path to baby cry audio file")
    parser.add_argument("voice_file", help="Path to human voice audio file")
    parser.add_argument("output_file", help="Path for output overlayed audio file")
    parser.add_argument("--cry_volume", type=float, default=1.0, help="Volume for cry audio (default: 1.0)")
    parser.add_argument("--voice_volume", type=float, default=1.0, help="Volume for voice audio (default: 1.0)")
    
    args = parser.parse_args()
    overlay_audio(args.cry_file, args.voice_file, args.output_file, args.cry_volume, args.voice_volume)