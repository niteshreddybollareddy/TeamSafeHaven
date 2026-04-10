"""
SILICON-9: CryNet-Micro-v2 Training Script
Binary Classification: Baby Cry vs Unknown
Optimized for xG26 SoC deployment with <10K parameters
"""

import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from datetime import datetime
import matplotlib.pyplot as plt

from dataset import get_dataloaders, load_data
import config
from model import CryNetMicroV2, count_parameters


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        return optimizer.learning_rate(optimizer.iterations).numpy()
    return float(optimizer.learning_rate.numpy())


@tf.function
def train_step(model, inputs, labels, criterion, optimizer):
    """Single training step"""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, outputs)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, outputs


def train_epoch(model, loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} Train', ncols=100, leave=False)
    for inputs, labels in pbar:
        loss, outputs = train_step(model, inputs, labels, criterion, optimizer)

        total_loss += loss.numpy()
        num_batches += 1

        preds = tf.argmax(outputs, axis=1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

        pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, f1, cm


def validate(model, loader, criterion, return_probs=False):
    """Validate the model"""
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    num_batches = 0

    pbar = tqdm(loader, desc='Validation', ncols=100, leave=False)
    for inputs, labels in pbar:
        outputs = model(inputs, training=False)
        loss = criterion(labels, outputs)

        total_loss += loss.numpy()
        num_batches += 1

        probs = outputs.numpy()  # Already softmax from model
        preds = tf.argmax(outputs, axis=1).numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    # Binary ROC-AUC
    all_probs_arr = np.array(all_probs)[:, 1]  # P(Unknown)
    try:
        auc_score = roc_auc_score(all_labels, all_probs_arr)
    except:
        auc_score = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if return_probs:
        return avg_loss, accuracy, f1, auc_score, cm, all_labels, all_probs
    return avg_loss, accuracy, f1, auc_score, cm


def save_training_graphs(history, graph_dir, test_labels=None, test_probs=None):
    """Save training visualization graphs"""
    os.makedirs(graph_dir, exist_ok=True)

    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(graph_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(graph_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # F1 curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_f1'], label='Train F1', linewidth=2)
    plt.plot(history['val_f1'], label='Val F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(graph_dir, 'f1_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(history['lr'], linewidth=2, color='#e74c3c')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule (Cosine Decay)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(graph_dir, 'lr_schedule.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC and PR curves (if test data provided)
    if test_labels is not None and test_probs is not None:
        test_probs_arr = np.array(test_probs)
        test_probs_binary = test_probs_arr[:, 1]  # P(Unknown)

        # ROC Curve
        fpr, tpr, _ = roc_curve(test_labels, test_probs_binary)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2, color='#3498db')
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Binary Classification', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(graph_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(test_labels, test_probs_binary)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})', linewidth=2, color='#2ecc71')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(graph_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    graph_dir = os.path.join(config.MODEL_DIR, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.LOG_DIR, f'training_log_{timestamp}.txt')
    logger = Logger(log_file)
    sys.stdout = logger

    # Device check
    gpus = tf.config.list_physical_devices('GPU')
    device_name = 'GPU' if gpus else 'CPU'

    print(f'='*70)
    print(f'SILICON-9: CRYNET-MICRO-V2 TRAINING')
    print(f'='*70)
    print(f'Training Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Using device: {device_name}')
    print(f'='*70)
    print(f'\nARCHITECTURE HIGHLIGHTS:')
    print(f'  ✓ Pure 2D CNN (no Transformer, no attention)')
    print(f'  ✓ All TFLite-compatible operations')
    print(f'  ✓ Target: <10K parameters')
    print(f'  ✓ Target: <20ms inference on xG26 @ 78MHz')
    print(f'  ✓ Binary classification: Baby Cry vs Unknown')
    print(f'  ✓ 1-second audio window (16kHz, 64 mel bins)')
    print(f'='*70)

    # Load data and create loaders
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    # Get sizes
    train_gen, _, _, _ = load_data()
    train_size = len(train_gen)

    # Calculate class distribution
    train_labels = train_gen.labels
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f'\nClass distribution:')
    for cls, count in zip(unique, counts):
        print(f'  Class {cls} ({config.CLASS_NAMES[cls]}): {count:,} samples ({count/len(train_labels)*100:.1f}%)')

    # Save stats
    np.save(os.path.join(config.MODEL_DIR, 'stats.npy'), stats)
    print(f'\nSaved normalization stats:')
    print(f'  Mean: {stats["mean"]:.6f}')
    print(f'  Std:  {stats["std"]:.6f}')

    # Initialize CryNet-Micro-v2
    model = CryNetMicroV2(
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )

    # Build model with dummy input (1s audio = 98 time frames)
    dummy_input = tf.zeros((1, 1, 64, 98))
    model(dummy_input)

    total_params = count_parameters(model)
    print(f'\nModel Configuration:')
    print(f'  Architecture: CryNet-Micro-v2')
    print(f'  CNN Filters: {config.CNN_FILTERS}')
    print(f'  Dropout: {config.DROPOUT}')
    print(f'  Input shape: (batch, 1, 64, 98)')
    print(f'  Output shape: (batch, {config.NUM_CLASSES})')
    print(f'\nModel parameters: {total_params:,}')

    if total_params > config.MAX_PARAMS:
        print(f'❌ WARNING: Model has {total_params:,} parameters, exceeds limit of {config.MAX_PARAMS:,}')
        print(f'   Over budget by {total_params - config.MAX_PARAMS:,} parameters')
    else:
        print(f'✓ Within parameter budget!')
        print(f'  Remaining: {config.MAX_PARAMS - total_params:,} parameters')

    # Loss: Categorical Cross-Entropy with Label Smoothing
    criterion = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,  # Model outputs softmax
        label_smoothing=config.LABEL_SMOOTHING
    )

    # Optimizer: AdamW with Cosine Decay
    steps_per_epoch = train_size // config.BATCH_SIZE
    total_steps = config.NUM_EPOCHS * steps_per_epoch

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=total_steps,
        alpha=0.0  # Decay to 0
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.WEIGHT_DECAY
    )

    print(f'\nTraining Configuration:')
    print(f'  Batch size: {config.BATCH_SIZE}')
    print(f'  Initial learning rate: {config.LEARNING_RATE}')
    print(f'  Total epochs: {config.NUM_EPOCHS}')
    print(f'  Steps per epoch: {steps_per_epoch}')
    print(f'  Early stopping patience: {config.PATIENCE}')
    print(f'  Weight decay: {config.WEIGHT_DECAY}')
    print(f'  Label smoothing: {config.LABEL_SMOOTHING}')

    # Training loop
    best_f1 = 0
    best_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []
    }

    print(f'\n{"="*70}')
    print('Starting Training Loop')
    print(f'{"="*70}')

    for epoch in range(config.NUM_EPOCHS):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print(f'{"="*70}')

        # Train
        train_loss, train_acc, train_f1, train_cm = train_epoch(
            model, train_loader, criterion, optimizer, epoch+1
        )

        # Validate
        val_loss, val_acc, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion
        )

        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(get_lr(optimizer))

        print(f'\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}')
        print(f'Train Confusion Matrix:')
        print(f'  {config.CLASS_NAMES[0]:12s} {config.CLASS_NAMES[1]:12s}')
        for i, row in enumerate(train_cm):
            print(f'  {config.CLASS_NAMES[i]:12s} {row[0]:5d} {row[1]:5d}')

        print(f'\nVal   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
        print(f'Val Confusion Matrix:')
        print(f'  {config.CLASS_NAMES[0]:12s} {config.CLASS_NAMES[1]:12s}')
        for i, row in enumerate(val_cm):
            print(f'  {config.CLASS_NAMES[i]:12s} {row[0]:5d} {row[1]:5d}')

        print(f'\nLR: {get_lr(optimizer):.6f}')

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            patience_counter = 0

            # Save model weights
            model.save_weights(os.path.join(config.MODEL_DIR, 'best_model.weights.h5'))

            # Save metadata
            meta_data = {
                'epoch': epoch + 1,
                'val_f1': float(val_f1),
                'val_acc': float(val_acc),
                'val_auc': float(val_auc),
                'stats': stats,
                'config': {
                    'num_classes': config.NUM_CLASSES,
                    'cnn_filters': config.CNN_FILTERS,
                    'dropout': config.DROPOUT
                }
            }
            np.save(os.path.join(config.MODEL_DIR, 'meta_data.npy'), meta_data)
            print(f'✓ Saved best model (Epoch {epoch+1}): F1={best_f1:.4f}, Acc={best_acc:.4f}')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{config.PATIENCE}')

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break

    # Test evaluation
    print(f'\n{"="*70}')
    print('Testing Best Model')
    print(f'{"="*70}')

    try:
        model.load_weights(os.path.join(config.MODEL_DIR, 'best_model.weights.h5'))
        print('✓ Loaded best model weights')
    except Exception as e:
        print(f"❌ Error loading weights: {e}")

    test_loss, test_acc, test_f1, test_auc, test_cm, test_labels, test_probs = validate(
        model, test_loader, criterion, return_probs=True
    )

    print(f'\nFinal Test Results:')
    print(f'  Loss:     {test_loss:.4f}')
    print(f'  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f'  F1 Score: {test_f1:.4f}')
    print(f'  AUC:      {test_auc:.4f}')

    print(f'\nTest Confusion Matrix:')
    print(f'  {"":12s} {config.CLASS_NAMES[0]:12s} {config.CLASS_NAMES[1]:12s}')
    for i, row in enumerate(test_cm):
        print(f'  {config.CLASS_NAMES[i]:12s} {row[0]:5d} {row[1]:5d}')

    # Per-class metrics
    print(f'\nPer-Class Test Metrics:')
    for class_idx in range(config.NUM_CLASSES):
        class_name = config.CLASS_NAMES[class_idx]

        # Precision: TP / (TP + FP)
        tp = test_cm[class_idx][class_idx]
        fp = test_cm[:, class_idx].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall: TP / (TP + FN)
        fn = test_cm[class_idx, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score
        f1_class = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'  {class_name}:')
        print(f'    Precision: {precision:.4f}')
        print(f'    Recall:    {recall:.4f}')
        print(f'    F1-Score:  {f1_class:.4f}')

    # Save training graphs
    save_training_graphs(history, graph_dir, test_labels, test_probs)
    print(f'\n✓ Graphs saved to: {graph_dir}')

    print(f'\n{"="*70}')
    print(f'Training Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best Val F1: {best_f1:.4f}, Best Val Acc: {best_acc:.4f}')
    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')
    print(f'Log saved to: {log_file}')
    print(f'{"="*70}')

    logger.close()
    sys.stdout = sys.stdout.terminal


if __name__ == '__main__':
    main()


"""
================================================================================
SILICON-9 TRAINING PIPELINE - 1-SECOND AUDIO WINDOWS
================================================================================

OVERVIEW:
Binary classification training for Baby Cry vs Unknown sounds using CryNet-
Micro-v2, optimized for xG26 SoC deployment with strict parameter budget.

================================================================================
KEY DESIGN DECISIONS:
================================================================================

1. AUDIO WINDOW: 1 second (16,000 samples @ 16kHz)
   Rationale: 72% of baby cry files naturally fit in 1s window
   - Minimizes padding bias
   - Unknown sounds chunked into 1s segments
   - Matches MLTK preprocessing output: (64, 98) mel-spectrogram

2. LOSS FUNCTION: Categorical Cross-Entropy with Label Smoothing (0.1)
   Rationale: Better than Focal Loss for balanced binary classification
   - Label smoothing prevents overconfident predictions
   - More stable for INT8 quantization
   - Softmax output (not sigmoid) for 3-class scalability

3. OPTIMIZER: AdamW with Cosine Decay
   Rationale: Simple, effective, no warmup needed
   - Cosine decay smoothly reduces LR to 0
   - Weight decay (0.0001) for L2 regularization
   - No warmup (CNN is stable, not Transformer)

4. PARAMETER BUDGET: <10,000 parameters
   Rationale: xG26 SoC arena memory constraints
   - CryNet-Micro-v2: 6,980 parameters ✓
   - All TFLite-compatible operations
   - 12-18ms inference target @ 78MHz

================================================================================
ARCHITECTURE: CryNet-Micro-v2
================================================================================

Input: (batch, 1, 64, 98) - 1s mel-spectrogram
  ↓ Transpose to (batch, 64, 98, 1) for Conv2D

Block 1: Conv2D(6) + BN + ReLU + MaxPool(2x2)
  → (batch, 32, 49, 6)

Block 2: Conv2D(12) + BN + ReLU + MaxPool(2x2)
  → (batch, 16, 24, 12)

Block 3: Conv2D(18) + BN + ReLU + MaxPool(2x2)
  → (batch, 8, 12, 18)

Block 4: Conv2D(24) + BN + ReLU + GlobalAvgPool
  → (batch, 24)

Head: Dense(12) + ReLU + Dropout(0.4) + Dense(2) + Softmax
  → (batch, 2)

Total Parameters: 6,980 (3,020 under budget)

================================================================================
TFLITE OPERATIONS (ALL SUPPORTED ON XG26):
================================================================================

✓ Conv2D              - 2D Convolution
✓ BatchNormalization  - Fused with Conv2D in TFLite
✓ ReLU                - Activation
✓ MaxPooling2D        - Downsampling
✓ GlobalAveragePooling2D - Spatial reduction
✓ Dense               - Fully connected
✓ Dropout             - Ignored in inference
✓ Softmax             - Output activation

NO unsupported operations (vs SILICON-8 which had 2 unsupported ops)

================================================================================
TRAINING FEATURES:
================================================================================

• CATEGORICAL CROSS-ENTROPY with Label Smoothing (0.1)
  - Prevents overconfident predictions
  - Better for INT8 quantization stability

• COSINE DECAY LR SCHEDULE
  - Smooth convergence from 1e-3 to 0
  - No warmup needed (simple CNN)

• EARLY STOPPING
  - Monitors validation F1 score
  - Patience: 20 epochs
  - Saves best model automatically

• AUGMENTATION (in dataset.py)
  - SpecAugment: Time + Frequency masking
  - Time shifting
  - Gaussian noise

================================================================================
BABY CRY ACOUSTIC PROFILE:
================================================================================

Fundamental Frequency: 300-600 Hz (vs dog: 200-400 Hz)
Harmonics: Strong up to 2-3 kHz
High-frequency energy: Up to 5-6 kHz
Duration: 0.3-0.8s bursts (natural crying patterns)
Spectral centroid: High (piercing, penetrating quality)

Model learns these discriminative features through:
- Low-level features (Block 1): Harmonics, F0
- Mid-level patterns (Block 2): Spectral envelopes
- High-level patterns (Block 3): Temporal structure
- Feature compression (Block 4): Cry vs non-cry discrimination

================================================================================
FUTURE SCALABILITY (3-CLASS):
================================================================================

When adding Dog Bark class:
1. Change config.NUM_CLASSES = 3
2. Change config.CLASS_NAMES = ['Baby Cry', 'Dog', 'Unknown']
3. Retrain model (or freeze CNN, retrain head only)
4. Softmax already supports multi-class (no code change)

Dog Bark Acoustic Profile:
- F0: 200-400 Hz (lower than baby)
- Less harmonic, more noise-like
- Sharper temporal edges (attack/decay)
- Shorter duration: 0.1-0.5s

Model will discriminate based on F0, harmonicity, and temporal structure.

================================================================================
DEPLOYMENT (XG26 SOC):
================================================================================

For inference:
1. Record 1s audio @ 16kHz mono
2. Extract mel-spectrogram: (64, 98) using MLTK
3. Normalize: (spec - mean) / std
4. Add dims: (1, 1, 64, 98)
5. Run TFLite model
6. Output: [P(Baby), P(Unknown)]
7. Threshold: P(Baby) > 0.5 → Baby Cry detected

Expected Performance:
- Inference time: 12-18ms @ 78MHz
- Memory: <10KB for parameters
- Arena: Minimal (all ops are xG26-compatible)

================================================================================
"""
