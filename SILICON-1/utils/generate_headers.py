import numpy as np
import config

# Copy the logic from preprocess.py to ensure 1:1 match
def hz_to_mel(frequencies):
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)

def mel_to_hz(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def generate_arrays():
    print(f"Generating C Header for:")
    print(f"Sample Rate: {config.SAMPLE_RATE}")
    print(f"FFT Size: {config.N_FFT}")
    print(f"Mel Bins: {config.N_MELS}")

    # -----------------------------
    # 1. Generate Hanning Window
    # -----------------------------
    # Note: In C, we often use float32.
    window = np.hanning(config.N_FFT).astype(np.float32)

    # -----------------------------
    # 2. Generate Mel Filterbank
    # -----------------------------
    sr = config.SAMPLE_RATE
    n_fft = config.N_FFT
    n_mels = config.N_MELS
    fmin = 50
    fmax = 8000

    fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filters = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    
    for i in range(n_mels):
        b_left = bin_points[i]
        b_center = bin_points[i + 1]
        b_right = bin_points[i + 2]
        
        for k in range(b_left, b_center):
            filters[i, k] = (k - b_left) / (b_center - b_left)
        for k in range(b_center, b_right):
            filters[i, k] = (b_right - k) / (b_right - b_center)

    # -----------------------------
    # 3. Write to .h file
    # -----------------------------
    with open("dsp_config.h", "w") as f:
        f.write("#ifndef DSP_CONFIG_H\n")
        f.write("#define DSP_CONFIG_H\n\n")
        
        f.write(f"#define N_FFT {config.N_FFT}\n")
        f.write(f"#define N_MELS {config.N_MELS}\n")
        f.write(f"#define FFT_BINS {filters.shape[1]}\n\n")

        # Write Window Array
        f.write("// Hanning Window (Applied before FFT)\n")
        f.write(f"const float hanning_window[{len(window)}] = {{\n")
        for i, val in enumerate(window):
            f.write(f"    {val:.6f}f, ")
            if (i + 1) % 10 == 0: f.write("\n")
        f.write("};\n\n")

        # Write Mel Filterbank
        # Note: We flatten the 2D matrix into a 1D array for C
        f.write("// Mel Filterbank Weights (Flattened 2D Matrix)\n")
        f.write("// Access logic: filter_bank[mel_idx * FFT_BINS + fft_bin_idx]\n")
        f.write(f"const float mel_filter_bank[{filters.size}] = {{\n")
        
        flat_filters = filters.flatten()
        for i, val in enumerate(flat_filters):
            f.write(f"    {val:.6f}f, ")
            if (i + 1) % 10 == 0: f.write("\n")
        f.write("};\n\n")
        
        f.write("#endif // DSP_CONFIG_H\n")

    print("✓ Successfully generated 'dsp_config.h'")
    print(f"  - Window Size: {len(window) * 4} bytes")
    print(f"  - Mel Matrix Size: {filters.size * 4} bytes")

if __name__ == "__main__":
    generate_arrays()