"""
SILICON-12 Training Script
3-Class Audio Classification: Baby Cry, Dog Barking, Unknown

Based on SILICON-9's proven training approach:
- CryNet3Class architecture (standard Conv2D, ~7.5K params)
- Fixed 98-frame spectrograms (1.5s audio)
- Categorical cross-entropy with label smoothing
- Cosine learning rate decay with warmup
- SpecAugment data augmentation
"""

import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders, get_class_weights, load_data
from model import CryNet3Class, count_parameters
import config


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate with linear warmup followed by cosine decay."""

    def __init__(self, initial_lr, warmup_steps, decay_steps, min_lr=1e-6):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup phase
        warmup_lr = (step / self.warmup_steps) * self.initial_lr

        # Cosine decay phase
        decay_progress = (step - self.warmup_steps) / self.decay_steps
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_progress))
        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)


def get_lr(optimizer):
    """Get current learning rate."""
    if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        return optimizer.learning_rate(optimizer.iterations).numpy()
    return float(optimizer.learning_rate.numpy())


@tf.function
def train_step(model, inputs, labels, loss_fn, optimizer):
    """Single training step."""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        # Convert labels to one-hot
        labels_onehot = tf.one_hot(labels, depth=config.NUM_CLASSES)
        loss = loss_fn(labels_onehot, outputs)

    gradients = tape.gradient(loss, model.trainable_variables)

    # Gradient clipping
    if config.GRAD_CLIP:
        gradients, _ = tf.clip_by_global_norm(gradients, config.GRAD_CLIP)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, outputs


def train_epoch(model, loader, loss_fn, optimizer, epoch):
    """Train for one epoch."""
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    for inputs, labels in loader:
        loss, outputs = train_step(model, inputs, labels, loss_fn, optimizer)

        total_loss += loss.numpy()
        num_batches += 1

        preds = tf.argmax(outputs, axis=-1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1


def validate(model, loader, loss_fn, return_probs=False):
    """Validate the model."""
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    num_batches = 0

    for inputs, labels in loader:
        outputs = model(inputs, training=False)
        labels_onehot = tf.one_hot(labels, depth=config.NUM_CLASSES)
        loss = loss_fn(labels_onehot, outputs)

        total_loss += loss.numpy()
        num_batches += 1

        probs = outputs.numpy()
        preds = tf.argmax(outputs, axis=-1).numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # AUC (macro-average one-vs-rest)
    try:
        auc_score = roc_auc_score(all_labels, np.array(all_probs),
                                   multi_class='ovr', average='macro')
    except:
        auc_score = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if return_probs:
        return avg_loss, accuracy, f1, auc_score, cm, all_labels, all_probs
    return avg_loss, accuracy, f1, auc_score, cm


def save_plots(history, save_dir):
    """Save training plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=150)
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=150)
    plt.close()

    # F1
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1.png'), dpi=150)
    plt.close()

    # Learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'lr.png'), dpi=150)
    plt.close()


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("SILICON-12: 3-CLASS AUDIO CLASSIFIER TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(config.EXPERIMENT_DIR, f'crynet3class_{timestamp}')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nExperiment: {exp_dir}")

    # Device
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Device: {'GPU' if gpus else 'CPU'}")

    # Load data
    print("\n" + "-"*70)
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    # Save stats
    np.save(os.path.join(checkpoint_dir, 'stats.npy'), stats)
    print(f"Stats saved: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Class distribution
    train_gen, _, _, _ = load_data()
    train_size = len(train_gen)
    print(f"\nTraining samples: {train_size}")

    # Class weights
    if config.USE_CLASS_WEIGHTS:
        class_weights = get_class_weights()
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None

    # Model
    print("\n" + "-"*70)
    print("Building model...")
    model = CryNet3Class(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)

    # Build with dummy input
    dummy = tf.zeros((1, 1, 64, config.N_TIME_FRAMES))
    _ = model(dummy)

    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    print(f"Budget: {config.MAX_PARAMS:,}")
    if params <= config.MAX_PARAMS:
        print(f"Within budget ({config.MAX_PARAMS - params:,} remaining)")
    else:
        print(f"EXCEEDS budget by {params - config.MAX_PARAMS:,}!")

    model.summary()

    # Loss function (categorical cross-entropy with label smoothing)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,  # Model outputs softmax
        label_smoothing=config.LABEL_SMOOTHING
    )
    print(f"\nLoss: CategoricalCrossentropy (smoothing={config.LABEL_SMOOTHING})")

    # Learning rate schedule
    steps_per_epoch = train_size // config.BATCH_SIZE
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
    decay_steps = (config.NUM_EPOCHS - config.WARMUP_EPOCHS) * steps_per_epoch

    lr_schedule = WarmupCosineDecay(
        initial_lr=config.LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr=config.MIN_LR
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.WEIGHT_DECAY
    )

    print(f"\nOptimizer: AdamW")
    print(f"  Initial LR: {config.LEARNING_RATE}")
    print(f"  Warmup: {config.WARMUP_EPOCHS} epochs")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")

    # Training
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []
    }

    best_f1 = 0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-"*40)

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, loss_fn, optimizer, epoch
        )

        # Validate
        val_loss, val_acc, val_f1, val_auc, val_cm = validate(
            model, val_loader, loss_fn
        )

        lr = get_lr(optimizer)

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(lr)

        print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f}")
        print(f"Val:   Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}")
        print(f"LR: {lr:.6f}")

        # Best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            np.save(os.path.join(checkpoint_dir, 'best_meta.npy'), {
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'stats': stats
            })
            print(f"Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.PATIENCE}")

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)

    model.load_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))

    test_loss, test_acc, test_f1, test_auc, test_cm, test_labels, test_probs = validate(
        model, test_loader, loss_fn, return_probs=True
    )

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC:      {test_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_cm)

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    test_preds = np.argmax(test_probs, axis=1)
    print(classification_report(test_labels, test_preds,
                                 target_names=config.CLASS_NAMES, digits=4))

    # Save plots
    save_plots(history, os.path.join(exp_dir, 'plots'))
    print(f"\nPlots saved to: {exp_dir}/plots")

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"\nModel saved to: {checkpoint_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
