"""
Quick comparison script between baseline and improved models.
Run this to see the architectural differences and parameter counts.
"""

import tensorflow as tf
import numpy as np

print("="*80)
print("MODEL COMPARISON: Baseline vs Improved")
print("="*80)

# Import both models
from Model import TransformerClassifier as BaselineModel
from Model_improved import ImprovedTransformerClassifier as ImprovedModel

# Baseline configuration (from Trained_2)
baseline_config = {
    'num_classes': 2,
    'd_model': 18,
    'num_heads': 3,
    'num_layers': 2,
    'dropout': 0.1
}

# Improved configuration
improved_config = {
    'num_classes': 2,
    'd_model': 18,
    'num_heads': 3,
    'num_layers': 3,
    'dropout': 0.15,
    'use_se_block': True,
    'se_reduction': 3
}

print("\n" + "="*80)
print("CONFIGURATION COMPARISON")
print("="*80)

print("\n{:<25} {:<25} {:<25}".format("Parameter", "Baseline", "Improved"))
print("-"*80)
print("{:<25} {:<25} {:<25}".format("d_model", baseline_config['d_model'], improved_config['d_model']))
print("{:<25} {:<25} {:<25}".format("num_heads", baseline_config['num_heads'], improved_config['num_heads']))
print("{:<25} {:<25} {:<25}".format("num_layers", baseline_config['num_layers'], improved_config['num_layers']))
print("{:<25} {:<25} {:<25}".format("dropout", baseline_config['dropout'], improved_config['dropout']))
print("{:<25} {:<25} {:<25}".format("CNN Architecture", "Single 3x3 (8 filters)", "Multi-scale 3x3+5x5+7x7"))
print("{:<25} {:<25} {:<25}".format("CNN Filters", "8", "6 (3+2+1)"))
print("{:<25} {:<25} {:<25}".format("SE Block", "No", "Yes (reduction=3)"))

# Build models
print("\n" + "="*80)
print("BUILDING MODELS...")
print("="*80)

baseline = BaselineModel(**baseline_config)
improved = ImprovedModel(**improved_config)

# Build with dummy input
dummy = tf.zeros((1, 1, 64, 128))
_ = baseline(dummy)
_ = improved(dummy)

baseline_params = baseline.count_params()
improved_params = improved.count_params()

print("\n{:<25} {:<25} {:<25}".format("Metric", "Baseline", "Improved"))
print("-"*80)
print("{:<25} {:<25,} {:<25,}".format("Total Parameters", baseline_params, improved_params))
print("{:<25} {:<25,} {:<25,}".format("Parameter Budget", 17000, 17000))
print("{:<25} {:<25,} {:<25,}".format("Remaining Budget", 17000 - baseline_params, 17000 - improved_params))
print("{:<25} {:<25,} {:<25,}".format("Additional Params", 0, improved_params - baseline_params))

# Calculate parameter increase
param_increase_pct = ((improved_params - baseline_params) / baseline_params) * 100
print("\nParameter Increase: +{:.1f}% ({:,} additional parameters)".format(
    param_increase_pct, improved_params - baseline_params))

# Test inference
print("\n" + "="*80)
print("TESTING INFERENCE")
print("="*80)

test_input = tf.random.normal((4, 1, 64, 128))

print("\nBaseline Model:")
baseline_out = baseline(test_input, training=False)
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {baseline_out.shape}")
print(f"  Output sample: {baseline_out[0].numpy()}")

print("\nImproved Model:")
improved_out = improved(test_input, training=False)
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {improved_out.shape}")
print(f"  Output sample: {improved_out[0].numpy()}")

# Performance expectations
print("\n" + "="*80)
print("PERFORMANCE EXPECTATIONS")
print("="*80)

baseline_performance = {
    'Test Accuracy': '95.98%',
    'Test F1': '0.9598',
    'Test AUC': '0.9942',
    'Train-Test Gap': '1.09%'
}

improved_expected = {
    'Test Accuracy': '98-99% (projected)',
    'Test F1': '0.980-0.990 (projected)',
    'Test AUC': '0.995+ (projected)',
    'Train-Test Gap': '<2% (low overfitting)'
}

print("\n{:<25} {:<30} {:<30}".format("Metric", "Baseline (Actual)", "Improved (Expected)"))
print("-"*80)
for key in baseline_performance:
    print("{:<25} {:<30} {:<30}".format(key, baseline_performance[key], improved_expected[key]))

# Key improvements
print("\n" + "="*80)
print("KEY ARCHITECTURAL IMPROVEMENTS")
print("="*80)

improvements = [
    ("Multi-Scale CNN", "3x3, 5x5, 7x7 kernels for diverse temporal patterns", "+1.0-1.5%"),
    ("Squeeze-Excitation", "Channel attention for frequency selection", "+0.3-0.5%"),
    ("Deeper Transformer", "3 layers instead of 2 for better capacity", "+0.5-1.0%"),
    ("Focal Loss", "Emphasizes hard examples during training", "+0.3-0.5%"),
    ("LR Warmup", "Stabilizes early training", "+0.2-0.3%"),
    ("Enhanced Augmentation", "Stronger regularization prevents overfitting", "+0.5-0.8%")
]

print("\n{:<25} {:<45} {:<15}".format("Improvement", "Description", "Expected Gain"))
print("-"*80)
for name, desc, gain in improvements:
    print("{:<25} {:<45} {:<15}".format(name, desc, gain))

print("\n{:<25} {:<45} {:<15}".format("TOTAL EXPECTED", "", "+3.0-5.0%"))

print("\n" + "="*80)
print("TRAINING RECOMMENDATIONS")
print("="*80)

print("""
1. Train the improved model:
   python3 train_improved.py

2. Monitor training in logs_improved/

3. Compare with baseline results in Trained_2/

4. Check graphs in models_improved/graphs/

5. Expected training time: ~10-15 minutes (250 epochs with early stopping)

6. Expected outcome: 98-99% test accuracy (vs 95.98% baseline)
""")

print("="*80)
print("COMPARISON COMPLETE")
print("="*80)
