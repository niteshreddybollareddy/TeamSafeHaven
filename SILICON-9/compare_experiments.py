"""
SILICON-9: Experiment Comparison Tool
Compare multiple training runs and identify best configuration
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


class ExperimentComparator:
    """Compare multiple training experiments"""

    def __init__(self, experiments_dir='experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.experiments = []

    def load_experiments(self):
        """Load all experiments from experiments directory"""

        if not self.experiments_dir.exists():
            print(f"❌ Experiments directory not found: {self.experiments_dir}")
            return

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            # Load experiment data
            config_path = exp_dir / 'config.json'
            history_path = exp_dir / 'logs' / 'history.json'
            test_results_path = exp_dir / 'logs' / 'test_results.json'
            best_metrics_path = exp_dir / 'logs' / 'best_metrics.json'

            if not all([config_path.exists(), history_path.exists(),
                       test_results_path.exists(), best_metrics_path.exists()]):
                print(f"⚠ Skipping incomplete experiment: {exp_dir.name}")
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            with open(history_path, 'r') as f:
                history = json.load(f)

            with open(test_results_path, 'r') as f:
                test_results = json.load(f)

            with open(best_metrics_path, 'r') as f:
                best_metrics = json.load(f)

            self.experiments.append({
                'name': exp_dir.name,
                'config': config,
                'history': history,
                'test_results': test_results,
                'best_metrics': best_metrics,
                'path': exp_dir
            })

        print(f"✓ Loaded {len(self.experiments)} experiments")

    def generate_comparison_table(self):
        """Generate comparison table of all experiments"""

        if not self.experiments:
            print("No experiments to compare")
            return None

        # Prepare data
        data = []
        for exp in self.experiments:
            row = {
                'Experiment': exp['name'],
                'Best Val F1': exp['best_metrics']['best_val_f1'],
                'Best Val Acc': exp['best_metrics']['best_val_acc'],
                'Best Val AUC': exp['best_metrics']['best_val_auc'],
                'Best Epoch': exp['best_metrics']['best_epoch'] + 1,
                'Test F1': exp['test_results']['f1'],
                'Test Acc': exp['test_results']['accuracy'],
                'Test AUC': exp['test_results']['auc'],
                'Test Precision': exp['test_results']['precision'],
                'Test Recall': exp['test_results']['recall'],
                'Total Params': exp['config']['model']['max_params'],
                'Batch Size': exp['config']['training']['batch_size'],
                'Learning Rate': exp['config']['training']['learning_rate'],
                'Dropout': exp['config']['model']['dropout'],
                'Label Smoothing': exp['config']['training']['label_smoothing']
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by test F1 score
        df = df.sort_values('Test F1', ascending=False)

        return df

    def print_comparison_table(self, df):
        """Print formatted comparison table"""

        print("\n" + "="*120)
        print("EXPERIMENT COMPARISON - SORTED BY TEST F1 SCORE")
        print("="*120)

        # Format for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 4)

        print(df.to_string(index=False))
        print("="*120)

        # Highlight best experiment
        best_exp = df.iloc[0]
        print(f"\n🏆 BEST EXPERIMENT: {best_exp['Experiment']}")
        print(f"   Test F1:  {best_exp['Test F1']:.4f}")
        print(f"   Test Acc: {best_exp['Test Acc']:.4f} ({best_exp['Test Acc']*100:.2f}%)")
        print(f"   Test AUC: {best_exp['Test AUC']:.4f}")
        print(f"   Trained for {best_exp['Best Epoch']:.0f} epochs")
        print()

    def statistical_comparison(self, df):
        """Perform statistical comparison between experiments"""

        print("\n" + "="*120)
        print("STATISTICAL ANALYSIS")
        print("="*120)

        # Get baseline (first alphabetically or explicitly named 'baseline')
        baseline_idx = 0
        for i, name in enumerate(df['Experiment']):
            if 'baseline' in name.lower():
                baseline_idx = i
                break

        baseline = df.iloc[baseline_idx]
        baseline_f1 = baseline['Test F1']

        print(f"\nBaseline: {baseline['Experiment']}")
        print(f"Baseline Test F1: {baseline_f1:.4f}")
        print("\nComparison to Baseline:")
        print("-" * 120)

        for i, row in df.iterrows():
            if i == baseline_idx:
                continue

            f1_diff = row['Test F1'] - baseline_f1
            f1_pct = (f1_diff / baseline_f1) * 100

            symbol = "✓✓✓" if abs(f1_diff) > 0.03 else "✓✓" if abs(f1_diff) > 0.015 else "✓" if abs(f1_diff) > 0.005 else "~"
            direction = "▲" if f1_diff > 0 else "▼" if f1_diff < 0 else "="

            print(f"{row['Experiment']:40s} | F1: {row['Test F1']:.4f} | "
                  f"{direction} {abs(f1_diff):+.4f} ({f1_pct:+.2f}%) {symbol}")

        print("\nLegend: ✓✓✓ Strong improvement (>3%), ✓✓ Moderate (1.5-3%), ✓ Small (0.5-1.5%), ~ Negligible (<0.5%)")
        print("="*120)

    def plot_comparison_charts(self):
        """Generate comparison visualizations"""

        if not self.experiments:
            return

        output_dir = Path('comparison_results')
        output_dir.mkdir(exist_ok=True)

        # 1. Test F1 Score Comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        exp_names = [exp['name'] for exp in self.experiments]
        test_f1 = [exp['test_results']['f1'] for exp in self.experiments]

        # Sort by F1 score
        sorted_indices = np.argsort(test_f1)[::-1]
        exp_names_sorted = [exp_names[i] for i in sorted_indices]
        test_f1_sorted = [test_f1[i] for i in sorted_indices]

        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(test_f1_sorted))]

        bars = ax.barh(exp_names_sorted, test_f1_sorted, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, test_f1_sorted)):
            ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=10)

        ax.set_xlabel('Test F1 Score', fontsize=13, fontweight='bold')
        ax.set_title('Experiment Comparison: Test F1 Score', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(min(test_f1_sorted) - 0.02, max(test_f1_sorted) + 0.05)

        plt.tight_layout()
        plt.savefig(output_dir / 'test_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Multi-metric comparison (heatmap)
        metrics_data = []
        for exp in self.experiments:
            metrics_data.append([
                exp['test_results']['accuracy'],
                exp['test_results']['f1'],
                exp['test_results']['precision'],
                exp['test_results']['recall'],
                exp['test_results']['auc']
            ])

        metrics_df = pd.DataFrame(
            metrics_data,
            index=[exp['name'] for exp in self.experiments],
            columns=['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
        )

        # Normalize each column to [0, 1] for heatmap
        metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())

        fig, ax = plt.subplots(figsize=(10, len(self.experiments) * 0.5 + 2))

        sns.heatmap(metrics_normalized, annot=metrics_df.values, fmt='.4f',
                   cmap='RdYlGn', center=0.5, cbar_kws={'label': 'Normalized Score'},
                   linewidths=1, linecolor='gray', ax=ax)

        ax.set_title('Multi-Metric Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Experiments', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'multi_metric_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Training curves comparison (for selected experiments)
        # Plot top 3 experiments
        top_3_exp = sorted(self.experiments, key=lambda x: x['test_results']['f1'], reverse=True)[:3]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors_list = ['#e74c3c', '#3498db', '#2ecc71']

        # Loss curves
        for exp, color in zip(top_3_exp, colors_list):
            epochs = range(1, len(exp['history']['val_loss']) + 1)
            axes[0].plot(epochs, exp['history']['val_loss'], label=exp['name'],
                        color=color, linewidth=2, marker='o', markersize=4, alpha=0.8)

        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        axes[0].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        for exp, color in zip(top_3_exp, colors_list):
            epochs = range(1, len(exp['history']['val_acc']) + 1)
            axes[1].plot(epochs, exp['history']['val_acc'], label=exp['name'],
                        color=color, linewidth=2, marker='s', markersize=4, alpha=0.8)

        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
        axes[1].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # F1 curves
        for exp, color in zip(top_3_exp, colors_list):
            epochs = range(1, len(exp['history']['val_f1']) + 1)
            axes[2].plot(epochs, exp['history']['val_f1'], label=exp['name'],
                        color=color, linewidth=2, marker='^', markersize=4, alpha=0.8)

        axes[2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Validation F1 Score', fontsize=11, fontweight='bold')
        axes[2].set_title('Validation F1 Comparison', fontsize=13, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Comparison charts saved to: {output_dir}")

    def generate_report(self):
        """Generate comprehensive comparison report"""

        output_dir = Path('comparison_results')
        output_dir.mkdir(exist_ok=True)

        report = []
        report.append("="*100)
        report.append("SILICON-9 EXPERIMENT COMPARISON REPORT")
        report.append("="*100)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Experiments: {len(self.experiments)}")
        report.append("")

        # Get comparison table
        df = self.generate_comparison_table()

        if df is not None:
            # Best experiment
            best = df.iloc[0]
            report.append("BEST EXPERIMENT")
            report.append("-" * 100)
            report.append(f"Name:           {best['Experiment']}")
            report.append(f"Test F1:        {best['Test F1']:.4f}")
            report.append(f"Test Accuracy:  {best['Test Acc']:.4f} ({best['Test Acc']*100:.2f}%)")
            report.append(f"Test Precision: {best['Test Precision']:.4f}")
            report.append(f"Test Recall:    {best['Test Recall']:.4f}")
            report.append(f"Test AUC:       {best['Test AUC']:.4f}")
            report.append(f"Best Epoch:     {best['Best Epoch']:.0f}")
            report.append(f"Parameters:     {best['Total Params']:,}")
            report.append("")

            # Configuration
            best_exp_full = [exp for exp in self.experiments if exp['name'] == best['Experiment']][0]
            report.append("BEST CONFIGURATION")
            report.append("-" * 100)
            report.append(f"Dropout:        {best['Dropout']}")
            report.append(f"Label Smoothing: {best['Label Smoothing']}")
            report.append(f"Learning Rate:  {best['Learning Rate']}")
            report.append(f"Batch Size:     {best['Batch Size']:.0f}")

            aug_config = best_exp_full['config']['augmentation']
            report.append(f"Augmentation:")
            report.append(f"  Freq Mask:    {aug_config['freq_mask_param']}")
            report.append(f"  Time Mask:    {aug_config['time_mask_param']}")
            report.append(f"  Time Shift:   {aug_config['time_shift_ms']}ms")
            report.append(f"  Noise Std:    {aug_config['noise_std']}")
            report.append("")

            # Top 5 experiments
            report.append("TOP 5 EXPERIMENTS")
            report.append("-" * 100)
            report.append(f"{'Rank':<6} {'Experiment':<40} {'Test F1':<10} {'Test Acc':<10} {'Test AUC':<10}")
            report.append("-" * 100)

            for i, row in df.head(5).iterrows():
                rank = df.index.get_loc(i) + 1
                report.append(f"{rank:<6} {row['Experiment']:<40} {row['Test F1']:<10.4f} "
                            f"{row['Test Acc']:<10.4f} {row['Test AUC']:<10.4f}")

            report.append("")

            # Insights
            report.append("KEY INSIGHTS")
            report.append("-" * 100)

            # Dropout effect
            dropout_exps = df[['Dropout', 'Test F1']].groupby('Dropout').mean()
            if len(dropout_exps) > 1:
                best_dropout = dropout_exps['Test F1'].idxmax()
                report.append(f"✓ Best Dropout: {best_dropout:.2f} (Avg F1: {dropout_exps.loc[best_dropout, 'Test F1']:.4f})")

            # Label smoothing effect
            ls_exps = df[['Label Smoothing', 'Test F1']].groupby('Label Smoothing').mean()
            if len(ls_exps) > 1:
                best_ls = ls_exps['Test F1'].idxmax()
                report.append(f"✓ Best Label Smoothing: {best_ls:.2f} (Avg F1: {ls_exps.loc[best_ls, 'Test F1']:.4f})")

            # Performance statistics
            report.append(f"✓ F1 Score Range: {df['Test F1'].min():.4f} - {df['Test F1'].max():.4f}")
            report.append(f"✓ Accuracy Range: {df['Test Acc'].min():.4f} - {df['Test Acc'].max():.4f}")
            report.append(f"✓ Mean F1 Score: {df['Test F1'].mean():.4f} ± {df['Test F1'].std():.4f}")

            report.append("")

        report.append("="*100)

        # Save report
        report_text = "\n".join(report)
        print("\n" + report_text)

        with open(output_dir / 'comparison_report.txt', 'w') as f:
            f.write(report_text)

        # Save CSV
        if df is not None:
            df.to_csv(output_dir / 'experiments_comparison.csv', index=False)
            print(f"\n✓ Report saved to: {output_dir}/comparison_report.txt")
            print(f"✓ CSV saved to: {output_dir}/experiments_comparison.csv")

    def run_full_comparison(self):
        """Run complete comparison pipeline"""

        self.load_experiments()

        if not self.experiments:
            print("No experiments found. Run some training experiments first!")
            return

        # Generate comparison table
        df = self.generate_comparison_table()

        if df is not None:
            # Print table
            self.print_comparison_table(df)

            # Statistical comparison
            self.statistical_comparison(df)

            # Generate plots
            self.plot_comparison_charts()

            # Generate report
            self.generate_report()


def main():
    comparator = ExperimentComparator()
    comparator.run_full_comparison()


if __name__ == '__main__':
    main()
