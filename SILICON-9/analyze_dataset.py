"""
SILICON-9: Advanced Dataset Analysis Script
Deep dive into audio characteristics before training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import config

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


class DatasetAnalyzer:
    """Comprehensive dataset analysis for audio classification"""

    def __init__(self, baby_dir, unknown_dir):
        self.baby_dir = Path(baby_dir)
        self.unknown_dir = Path(unknown_dir)
        self.analysis_dir = Path('analysis_results')
        self.analysis_dir.mkdir(exist_ok=True)

        self.baby_stats = defaultdict(list)
        self.unknown_stats = defaultdict(list)

    def analyze_audio_file(self, audio_path):
        """Extract comprehensive statistics from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]

            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=64,
                fmin=50, fmax=8000,
                n_fft=512, hop_length=160, win_length=400
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Fundamental frequency (F0) estimation
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=50, fmax=1000, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]

            stats = {
                'duration': duration,
                'sr': sr,
                'samples': len(audio),

                # Spectral features
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'spectral_centroid_std': np.std(spectral_centroid),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),

                # Temporal features
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr),
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),

                # Energy distribution
                'energy_low_freq': np.mean(mel_spec_db[:16, :]),  # 0-2kHz
                'energy_mid_freq': np.mean(mel_spec_db[16:48, :]),  # 2-6kHz
                'energy_high_freq': np.mean(mel_spec_db[48:, :]),  # 6-8kHz

                # F0 features
                'f0_mean': np.mean(f0_valid) if len(f0_valid) > 0 else 0,
                'f0_std': np.std(f0_valid) if len(f0_valid) > 0 else 0,
                'f0_min': np.min(f0_valid) if len(f0_valid) > 0 else 0,
                'f0_max': np.max(f0_valid) if len(f0_valid) > 0 else 0,
                'voicing_rate': np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0,

                # Mel spectrogram
                'mel_spec': mel_spec_db
            }

            return stats

        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
            return None

    def analyze_class(self, audio_dir, class_name, max_files=None):
        """Analyze all files in a class directory"""
        # Search recursively for all .wav files
        audio_files = list(audio_dir.rglob('*.wav'))

        if max_files:
            audio_files = audio_files[:max_files]

        print(f"\n{'='*70}")
        print(f"Analyzing {class_name} class ({len(audio_files)} files)")
        print(f"{'='*70}")

        stats_dict = defaultdict(list)
        mel_specs = []

        for audio_path in tqdm(audio_files, desc=f"Processing {class_name}"):
            stats = self.analyze_audio_file(audio_path)

            if stats:
                # Store mel spectrogram separately
                mel_specs.append(stats['mel_spec'])
                del stats['mel_spec']

                # Store all other stats
                for key, value in stats.items():
                    stats_dict[key].append(value)

        return stats_dict, mel_specs

    def run_full_analysis(self, max_files_per_class=None):
        """Run comprehensive analysis on both classes"""

        print(f"\n{'#'*70}")
        print(f"# SILICON-9 DATASET ANALYSIS")
        print(f"# Scientific Analysis Before Training")
        print(f"{'#'*70}")

        # Analyze Baby Cry
        self.baby_stats, baby_mel_specs = self.analyze_class(
            self.baby_dir, 'Baby Cry', max_files_per_class
        )

        # Analyze Unknown
        self.unknown_stats, unknown_mel_specs = self.analyze_class(
            self.unknown_dir, 'Unknown', max_files_per_class
        )

        # Generate comparison report
        self.generate_comparison_report()

        # Generate visualizations
        self.generate_visualizations(baby_mel_specs, unknown_mel_specs)

        # Statistical significance tests
        self.statistical_tests()

        print(f"\n{'='*70}")
        print(f"✓ Analysis complete! Results saved to: {self.analysis_dir}")
        print(f"{'='*70}")

    def generate_comparison_report(self):
        """Generate detailed comparison report"""

        report = []
        report.append("="*80)
        report.append("SILICON-9 DATASET ANALYSIS REPORT")
        report.append("="*80)
        report.append("")

        # Sample counts
        report.append("1. DATASET SIZE")
        report.append("-" * 80)
        baby_count = len(self.baby_stats['duration'])
        unknown_count = len(self.unknown_stats['duration'])
        total_count = baby_count + unknown_count

        report.append(f"  Baby Cry:  {baby_count:,} samples")
        report.append(f"  Unknown:   {unknown_count:,} samples")
        report.append(f"  Total:     {total_count:,} samples")

        if total_count > 0:
            report.append(f"  Balance:   {baby_count / total_count * 100:.1f}% Baby Cry")
        else:
            report.append(f"  Balance:   N/A (no samples found)")
        report.append("")

        # Duration analysis
        report.append("2. AUDIO DURATION ANALYSIS")
        report.append("-" * 80)
        baby_durations = np.array(self.baby_stats['duration'])
        unknown_durations = np.array(self.unknown_stats['duration'])

        report.append(f"  Baby Cry:")
        report.append(f"    Mean:   {np.mean(baby_durations):.3f}s")
        report.append(f"    Median: {np.median(baby_durations):.3f}s")
        report.append(f"    Std:    {np.std(baby_durations):.3f}s")
        report.append(f"    Min:    {np.min(baby_durations):.3f}s")
        report.append(f"    Max:    {np.max(baby_durations):.3f}s")
        report.append(f"    <1s:    {np.sum(baby_durations < 1.0) / len(baby_durations) * 100:.1f}%")
        report.append(f"    <2s:    {np.sum(baby_durations < 2.0) / len(baby_durations) * 100:.1f}%")

        report.append(f"  Unknown:")
        report.append(f"    Mean:   {np.mean(unknown_durations):.3f}s")
        report.append(f"    Median: {np.median(unknown_durations):.3f}s")
        report.append(f"    Std:    {np.std(unknown_durations):.3f}s")
        report.append(f"    Min:    {np.min(unknown_durations):.3f}s")
        report.append(f"    Max:    {np.max(unknown_durations):.3f}s")
        report.append("")

        # Spectral analysis
        report.append("3. SPECTRAL CHARACTERISTICS")
        report.append("-" * 80)
        report.append(f"  Spectral Centroid (Hz):")
        report.append(f"    Baby Cry: {np.mean(self.baby_stats['spectral_centroid_mean']):.1f} ± {np.std(self.baby_stats['spectral_centroid_mean']):.1f}")
        report.append(f"    Unknown:  {np.mean(self.unknown_stats['spectral_centroid_mean']):.1f} ± {np.std(self.unknown_stats['spectral_centroid_mean']):.1f}")
        report.append(f"    Δ Difference: {np.mean(self.baby_stats['spectral_centroid_mean']) - np.mean(self.unknown_stats['spectral_centroid_mean']):.1f} Hz")
        report.append("")

        report.append(f"  Spectral Rolloff (Hz):")
        report.append(f"    Baby Cry: {np.mean(self.baby_stats['spectral_rolloff_mean']):.1f}")
        report.append(f"    Unknown:  {np.mean(self.unknown_stats['spectral_rolloff_mean']):.1f}")
        report.append("")

        # F0 analysis
        report.append("4. FUNDAMENTAL FREQUENCY (F0) ANALYSIS")
        report.append("-" * 80)
        baby_f0_valid = [f for f in self.baby_stats['f0_mean'] if f > 0]
        unknown_f0_valid = [f for f in self.unknown_stats['f0_mean'] if f > 0]

        report.append(f"  Baby Cry F0:")
        report.append(f"    Mean:   {np.mean(baby_f0_valid):.1f} Hz" if baby_f0_valid else "    Mean:   N/A")
        report.append(f"    Median: {np.median(baby_f0_valid):.1f} Hz" if baby_f0_valid else "    Median: N/A")
        report.append(f"    Range:  {np.min(baby_f0_valid):.1f} - {np.max(baby_f0_valid):.1f} Hz" if baby_f0_valid else "    Range:  N/A")

        report.append(f"  Unknown F0:")
        report.append(f"    Mean:   {np.mean(unknown_f0_valid):.1f} Hz" if unknown_f0_valid else "    Mean:   N/A")
        report.append(f"    Median: {np.median(unknown_f0_valid):.1f} Hz" if unknown_f0_valid else "    Median: N/A")
        report.append("")

        # Energy distribution
        report.append("5. FREQUENCY BAND ENERGY DISTRIBUTION")
        report.append("-" * 80)
        report.append(f"  Low Freq Energy (0-2kHz):")
        report.append(f"    Baby Cry: {np.mean(self.baby_stats['energy_low_freq']):.2f} dB")
        report.append(f"    Unknown:  {np.mean(self.unknown_stats['energy_low_freq']):.2f} dB")

        report.append(f"  Mid Freq Energy (2-6kHz):")
        report.append(f"    Baby Cry: {np.mean(self.baby_stats['energy_mid_freq']):.2f} dB")
        report.append(f"    Unknown:  {np.mean(self.unknown_stats['energy_mid_freq']):.2f} dB")

        report.append(f"  High Freq Energy (6-8kHz):")
        report.append(f"    Baby Cry: {np.mean(self.baby_stats['energy_high_freq']):.2f} dB")
        report.append(f"    Unknown:  {np.mean(self.unknown_stats['energy_high_freq']):.2f} dB")
        report.append("")

        # Voicing rate
        report.append("6. VOICING ANALYSIS")
        report.append("-" * 80)
        report.append(f"  Voicing Rate (% of frames with F0):")
        report.append(f"    Baby Cry: {np.mean(self.baby_stats['voicing_rate']) * 100:.1f}%")
        report.append(f"    Unknown:  {np.mean(self.unknown_stats['voicing_rate']) * 100:.1f}%")
        report.append("")

        # Key insights
        report.append("7. KEY DISCRIMINATIVE FEATURES")
        report.append("-" * 80)

        # Calculate effect sizes (Cohen's d)
        def cohens_d(x1, x2):
            n1, n2 = len(x1), len(x2)
            var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

        features_effect_size = []
        for feature in ['spectral_centroid_mean', 'spectral_rolloff_mean',
                       'energy_high_freq', 'voicing_rate', 'f0_mean']:
            if feature in self.baby_stats and feature in self.unknown_stats:
                baby_vals = [v for v in self.baby_stats[feature] if v > 0] if feature == 'f0_mean' else self.baby_stats[feature]
                unknown_vals = [v for v in self.unknown_stats[feature] if v > 0] if feature == 'f0_mean' else self.unknown_stats[feature]

                if baby_vals and unknown_vals:
                    d = abs(cohens_d(baby_vals, unknown_vals))
                    features_effect_size.append((feature, d))

        features_effect_size.sort(key=lambda x: x[1], reverse=True)

        report.append(f"  Most discriminative features (Cohen's d effect size):")
        for feature, d in features_effect_size[:5]:
            interpretation = "small" if d < 0.5 else "medium" if d < 0.8 else "large"
            report.append(f"    {feature:30s}: d={d:.3f} ({interpretation})")
        report.append("")

        # Training recommendations
        report.append("8. TRAINING RECOMMENDATIONS")
        report.append("-" * 80)

        # Check if 1s window is appropriate
        baby_under_1s = np.sum(baby_durations < 1.0) / len(baby_durations) * 100
        if baby_under_1s > 70:
            report.append(f"  ✓ 1-second window: {baby_under_1s:.1f}% of baby cries fit naturally")
        else:
            report.append(f"  ⚠ 1-second window: Only {baby_under_1s:.1f}% of baby cries fit")
            report.append(f"    Consider using longer window or cropping strategy")

        # Check class balance
        balance_ratio = len(self.baby_stats['duration']) / len(self.unknown_stats['duration'])
        if 0.9 <= balance_ratio <= 1.1:
            report.append(f"  ✓ Class balance: Well balanced ({balance_ratio:.2f}:1)")
        else:
            report.append(f"  ⚠ Class imbalance: {balance_ratio:.2f}:1 ratio")
            report.append(f"    Consider using class weights or focal loss")

        # Check feature discriminability
        max_effect_size = features_effect_size[0][1] if features_effect_size else 0
        if max_effect_size > 0.8:
            report.append(f"  ✓ Strong discriminative features detected (d={max_effect_size:.2f})")
            report.append(f"    Model should achieve high accuracy")
        elif max_effect_size > 0.5:
            report.append(f"  ✓ Moderate discriminative features (d={max_effect_size:.2f})")
            report.append(f"    Good accuracy expected with proper training")
        else:
            report.append(f"  ⚠ Weak discriminative features (d={max_effect_size:.2f})")
            report.append(f"    May need stronger augmentation or feature engineering")

        report.append("")
        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        print("\n" + report_text)

        with open(self.analysis_dir / 'dataset_analysis_report.txt', 'w') as f:
            f.write(report_text)

    def generate_visualizations(self, baby_mel_specs, unknown_mel_specs):
        """Generate comprehensive visualizations"""

        print("\nGenerating visualizations...")

        # 1. Duration distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(self.baby_stats['duration'], bins=50, alpha=0.7,
                    label='Baby Cry', color='#e74c3c', edgecolor='black')
        axes[0].axvline(1.0, color='green', linestyle='--', linewidth=2, label='1s window')
        axes[0].set_xlabel('Duration (seconds)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Baby Cry Duration Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(self.unknown_stats['duration'], bins=50, alpha=0.7,
                    label='Unknown', color='#3498db', edgecolor='black')
        axes[1].axvline(1.0, color='green', linestyle='--', linewidth=2, label='1s window')
        axes[1].set_xlabel('Duration (seconds)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Unknown Duration Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'duration_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Spectral centroid comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        baby_centroid = self.baby_stats['spectral_centroid_mean']
        unknown_centroid = self.unknown_stats['spectral_centroid_mean']

        positions = [1, 2]
        bp = ax.boxplot([baby_centroid, unknown_centroid], positions=positions,
                        widths=0.6, patch_artist=True,
                        labels=['Baby Cry', 'Unknown'])

        colors = ['#e74c3c', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Spectral Centroid (Hz)', fontsize=12)
        ax.set_title('Spectral Centroid Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'spectral_centroid_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. F0 distributions
        baby_f0_valid = [f for f in self.baby_stats['f0_mean'] if f > 0]
        unknown_f0_valid = [f for f in self.unknown_stats['f0_mean'] if f > 0]

        if baby_f0_valid and unknown_f0_valid:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(baby_f0_valid, bins=30, alpha=0.6, label='Baby Cry',
                   color='#e74c3c', edgecolor='black')
            ax.hist(unknown_f0_valid, bins=30, alpha=0.6, label='Unknown',
                   color='#3498db', edgecolor='black')

            ax.set_xlabel('Fundamental Frequency (Hz)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('F0 Distribution Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'f0_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Energy band comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        bands = ['Low\n(0-2kHz)', 'Mid\n(2-6kHz)', 'High\n(6-8kHz)']
        baby_energies = [
            np.mean(self.baby_stats['energy_low_freq']),
            np.mean(self.baby_stats['energy_mid_freq']),
            np.mean(self.baby_stats['energy_high_freq'])
        ]
        unknown_energies = [
            np.mean(self.unknown_stats['energy_low_freq']),
            np.mean(self.unknown_stats['energy_mid_freq']),
            np.mean(self.unknown_stats['energy_high_freq'])
        ]

        x = np.arange(len(bands))
        width = 0.35

        ax.bar(x - width/2, baby_energies, width, label='Baby Cry',
              color='#e74c3c', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, unknown_energies, width, label='Unknown',
              color='#3498db', alpha=0.8, edgecolor='black')

        ax.set_ylabel('Energy (dB)', fontsize=12)
        ax.set_title('Frequency Band Energy Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'energy_band_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Average mel spectrograms
        if baby_mel_specs and unknown_mel_specs:
            # Sample random spectrograms
            n_samples = min(100, len(baby_mel_specs), len(unknown_mel_specs))
            baby_sample = np.random.choice(len(baby_mel_specs), n_samples, replace=False)
            unknown_sample = np.random.choice(len(unknown_mel_specs), n_samples, replace=False)

            # Compute average (need to handle variable lengths)
            baby_avg = np.mean([baby_mel_specs[i] for i in baby_sample], axis=0)
            unknown_avg = np.mean([unknown_mel_specs[i] for i in unknown_sample], axis=0)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            im1 = axes[0].imshow(baby_avg, aspect='auto', origin='lower',
                                cmap='viridis', interpolation='bilinear')
            axes[0].set_xlabel('Time Frames', fontsize=12)
            axes[0].set_ylabel('Mel Frequency Bins', fontsize=12)
            axes[0].set_title('Average Baby Cry Mel-Spectrogram', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[0], label='dB')

            im2 = axes[1].imshow(unknown_avg, aspect='auto', origin='lower',
                                cmap='viridis', interpolation='bilinear')
            axes[1].set_xlabel('Time Frames', fontsize=12)
            axes[1].set_ylabel('Mel Frequency Bins', fontsize=12)
            axes[1].set_title('Average Unknown Mel-Spectrogram', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[1], label='dB')

            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'average_spectrograms.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("✓ Visualizations saved")

    def statistical_tests(self):
        """Perform statistical significance tests"""
        from scipy import stats

        print("\nPerforming statistical tests...")

        results = []
        results.append("\n" + "="*80)
        results.append("STATISTICAL SIGNIFICANCE TESTS")
        results.append("="*80)
        results.append("")

        # T-tests for key features
        features_to_test = [
            ('spectral_centroid_mean', 'Spectral Centroid'),
            ('spectral_rolloff_mean', 'Spectral Rolloff'),
            ('energy_high_freq', 'High Frequency Energy'),
            ('voicing_rate', 'Voicing Rate'),
            ('rms_mean', 'RMS Energy')
        ]

        for feature_key, feature_name in features_to_test:
            if feature_key in self.baby_stats and feature_key in self.unknown_stats:
                baby_vals = np.array(self.baby_stats[feature_key])
                unknown_vals = np.array(self.unknown_stats[feature_key])

                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(baby_vals, unknown_vals)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(baby_vals) + np.var(unknown_vals)) / 2)
                cohens_d = (np.mean(baby_vals) - np.mean(unknown_vals)) / pooled_std if pooled_std > 0 else 0

                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

                results.append(f"{feature_name}:")
                results.append(f"  t-statistic: {t_stat:.4f}")
                results.append(f"  p-value: {p_value:.6f} {significance}")
                results.append(f"  Cohen's d: {abs(cohens_d):.4f}")
                results.append("")

        results.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
        results.append("="*80)

        results_text = "\n".join(results)
        print(results_text)

        with open(self.analysis_dir / 'statistical_tests.txt', 'w') as f:
            f.write(results_text)


def main():
    # Initialize analyzer
    analyzer = DatasetAnalyzer(
        baby_dir='Dataset/Baby',
        unknown_dir='Dataset/Unknown'
    )

    # Run analysis (set max_files_per_class for quick test)
    # For full analysis, set max_files_per_class=None
    analyzer.run_full_analysis(max_files_per_class=None)


if __name__ == '__main__':
    main()
