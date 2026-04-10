import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataset import get_dataloaders, load_data
import config
from Model import ImprovedTransformerClassifier, count_parameters


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


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples.
        alpha: Balancing parameter (default: 0.25). Can be scalar or list for each class.
        from_logits: Whether input is logits (True) or probabilities (False).
    """
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth labels (B,) - integer class indices
            y_pred: Predictions (B, num_classes) - logits or probabilities
        """
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Get the predicted probability for the true class
        # Convert y_true to one-hot for gathering
        num_classes = tf.shape(y_pred)[-1]
        y_true_int = tf.cast(y_true, tf.int32)  # Ensure integer type for one_hot
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)

        # p_t: probability of true class
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Cross entropy: -log(p_t)
        ce = -tf.math.log(p_t)

        # Focal loss: alpha * (1 - p_t)^gamma * CE
        focal_loss = self.alpha * focal_weight * ce

        return tf.reduce_mean(focal_loss)


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.

    Args:
        initial_learning_rate: Peak learning rate after warmup
        warmup_steps: Number of steps for linear warmup
        decay_steps: Total steps for cosine decay (after warmup)
        alpha: Minimum learning rate as fraction of initial_lr (default: 0.0)
    """
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha=0.0, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Linear warmup
        warmup_lr = (step / warmup_steps) * self.initial_learning_rate

        # Cosine decay
        completed_fraction = (step - warmup_steps) / decay_steps
        cosine_decayed = 0.5 * (1.0 + tf.cos(np.pi * completed_fraction))
        decay_lr = (1.0 - self.alpha) * cosine_decayed + self.alpha
        decay_lr = self.initial_learning_rate * decay_lr

        # Choose warmup or decay based on current step
        lr = tf.cond(
            step < warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )

        return lr

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha
        }


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        return optimizer.learning_rate(optimizer.iterations).numpy()
    return float(optimizer.learning_rate.numpy())


@tf.function
def train_step(model, inputs, labels, criterion, optimizer, grad_clip=None):
    """Single training step with gradient clipping"""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, outputs)

    gradients = tape.gradient(loss, model.trainable_variables)

    if grad_clip:
        gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, outputs


def train_epoch(model, loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    grad_clip = getattr(config, 'GRAD_CLIP', None)

    pbar = tqdm(loader, desc=f'Epoch {epoch} Train', ncols=100, leave=False)
    for inputs, labels in pbar:
        loss, outputs = train_step(model, inputs, labels, criterion, optimizer, grad_clip)

        total_loss += loss.numpy()
        num_batches += 1

        preds = tf.argmax(outputs, axis=1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

        pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
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

        probs = tf.nn.softmax(outputs, axis=1).numpy()
        preds = tf.argmax(outputs, axis=1).numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # ROC-AUC
    if config.NUM_CLASSES == 2:
        all_probs_arr = np.array(all_probs)[:, 1]
        try:
            auc_score = roc_auc_score(all_labels, all_probs_arr)
        except:
            auc_score = 0.0
    else:
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
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # F1 curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, 'f1_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Warmup + Cosine Decay)')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, 'lr_schedule.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC and PR curves (if test data provided)
    if test_labels is not None and test_probs is not None and config.NUM_CLASSES == 2:
        test_probs_arr = np.array(test_probs)[:, 1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(test_labels, test_probs_arr)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(graph_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(test_labels, test_probs_arr)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
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

    print(f'='*60)
    print(f'PRODUCTION-SCALE IMPROVED TRAINING')
    print(f'='*60)
    print(f'Training Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Using device: {device_name}')
    print(f'='*60)
    print(f'\nIMPROVEMENTS:')
    print(f'  ✓ Multi-scale CNN (3x3 + 5x5 + 7x7 kernels)')
    print(f'  ✓ Squeeze-and-Excitation blocks')
    print(f'  ✓ Deeper Transformer ({config.NUM_LAYERS} layers)')
    print(f'  ✓ Warmup + Cosine LR schedule')
    print(f'  ✓ Focal Loss (gamma={config.FOCAL_GAMMA}, alpha={config.FOCAL_ALPHA})')
    print(f'  ✓ Enhanced augmentation (MixUp α={config.MIXUP_ALPHA})')
    print(f'='*60)

    # Load data and create loaders
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    # Get sizes
    train_gen, _, _, _ = load_data()
    train_size = len(train_gen)

    # Calculate class weights
    train_labels = train_gen.labels
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f'\nClass distribution:')
    for cls, count in zip(unique, counts):
        print(f' Class {cls} ({config.CLASS_NAMES[cls]}): {count} samples ({count/len(train_labels)*100:.1f}%)')

    # Save stats
    np.save(os.path.join(config.MODEL_DIR, 'stats.npy'), stats)
    print(f'\nSaved normalization stats - Mean: {stats["mean"]:.4f}, Std: {stats["std"]:.4f}')

    # Initialize improved model
    model = ImprovedTransformerClassifier(
        num_classes=config.NUM_CLASSES,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_se_block=config.USE_SE_BLOCK,
        se_reduction=config.SE_REDUCTION
    )

    # Build model with dummy input
    dummy_input = tf.zeros((1, 1, 64, 128))
    model(dummy_input)

    total_params = count_parameters(model)
    print(f'\nModel Configuration:')
    print(f'  d_model: {config.D_MODEL}')
    print(f'  num_heads: {config.NUM_HEADS}')
    print(f'  num_layers: {config.NUM_LAYERS}')
    print(f'  dropout: {config.DROPOUT}')
    print(f'  SE Block: {config.USE_SE_BLOCK}')
    print(f'\nModel parameters: {total_params:,}')

    if hasattr(config, 'MAX_PARAMS'):
        if total_params > config.MAX_PARAMS:
            print(f'WARNING: Model has {total_params:,} parameters, exceeds limit of {config.MAX_PARAMS:,}')
        else:
            print(f'✓ Within parameter budget ({config.MAX_PARAMS - total_params:,} remaining)')

    # Focal Loss (production-scale loss function)
    criterion = FocalLoss(
        gamma=config.FOCAL_GAMMA,
        alpha=config.FOCAL_ALPHA,
        from_logits=True
    )

    # Warmup + Cosine Decay schedule
    steps_per_epoch = train_size // config.BATCH_SIZE
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
    decay_steps = (config.NUM_EPOCHS - config.WARMUP_EPOCHS) * steps_per_epoch

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=0.0
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.WEIGHT_DECAY
    )

    print(f'\nTraining Configuration:')
    print(f'  Batch size: {config.BATCH_SIZE}')
    print(f'  Learning rate: {config.LEARNING_RATE}')
    print(f'  Warmup epochs: {config.WARMUP_EPOCHS}')
    print(f'  Total epochs: {config.NUM_EPOCHS}')
    print(f'  Early stopping patience: {config.PATIENCE}')
    print(f'  Gradient clipping: {config.GRAD_CLIP}')
    print(f'  Weight decay: {config.WEIGHT_DECAY}')

    # Training loop
    best_f1 = 0
    best_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []
    }

    print(f'\n{"="*60}')
    print('Starting Training Loop')
    print(f'{"="*60}')

    for epoch in range(config.NUM_EPOCHS):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print(f'{"="*60}')

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
        print(f'Train Confusion Matrix:\n{train_cm}')
        print(f'\nVal  - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
        print(f'Val Confusion Matrix:\n{val_cm}')
        print(f'LR: {get_lr(optimizer):.6f}')

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            patience_counter = 0

            # Save model weights
            model.save_weights(os.path.join(config.MODEL_DIR, 'best_model.weights.h5'))

            # Save metadata
            meta_data = {
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'stats': stats,
                'config': {
                    'd_model': config.D_MODEL,
                    'num_heads': config.NUM_HEADS,
                    'num_layers': config.NUM_LAYERS,
                    'dropout': config.DROPOUT
                }
            }
            np.save(os.path.join(config.MODEL_DIR, 'meta_data.npy'), meta_data)
            print(f'✓ Saved best model with F1: {best_f1:.4f}, Acc: {best_acc:.4f}')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{config.PATIENCE}')

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break

    # Test evaluation
    print(f'\n{"="*60}')
    print('Testing Best Model')
    print(f'{"="*60}')

    try:
        model.load_weights(os.path.join(config.MODEL_DIR, 'best_model.weights.h5'))
    except Exception as e:
        print(f"Error loading weights: {e}")

    test_loss, test_acc, test_f1, test_auc, test_cm, test_labels, test_probs = validate(
        model, test_loader, criterion, return_probs=True
    )

    print(f'\nFinal Test Results:')
    print(f' Loss:   {test_loss:.4f}')
    print(f' Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f' F1 Score: {test_f1:.4f}')
    print(f' AUC:    {test_auc:.4f}')
    print(f'\nTest Confusion Matrix:')
    print(test_cm)

    # Per-class metrics
    print(f'\nPer-Class Test Metrics:')
    if test_cm.shape[0] == 2:
        cry_precision = test_cm[0][0] / (test_cm[0][0] + test_cm[1][0]) if (test_cm[0][0] + test_cm[1][0]) > 0 else 0
        cry_recall = test_cm[0][0] / (test_cm[0][0] + test_cm[0][1]) if (test_cm[0][0] + test_cm[0][1]) > 0 else 0
        dog_precision = test_cm[1][1] / (test_cm[0][1] + test_cm[1][1]) if (test_cm[0][1] + test_cm[1][1]) > 0 else 0
        dog_recall = test_cm[1][1] / (test_cm[1][0] + test_cm[1][1]) if (test_cm[1][0] + test_cm[1][1]) > 0 else 0

        print(f' Baby Cry class:')
        print(f'  Precision: {cry_precision:.4f}')
        print(f'  Recall:  {cry_recall:.4f}')
        print(f' Dog class:')
        print(f'  Precision: {dog_precision:.4f}')
        print(f'  Recall:  {dog_recall:.4f}')

    # Save training graphs
    save_training_graphs(history, graph_dir, test_labels, test_probs)
    print(f'\nGraphs saved to: {graph_dir}')

    print(f'\n{"="*60}')
    print(f'Training Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best Val F1: {best_f1:.4f}, Best Val Acc: {best_acc:.4f}')
    print(f'Log saved to: {log_file}')
    print(f'{"="*60}')

    logger.close()
    sys.stdout = sys.stdout.terminal


if __name__ == '__main__':
    main()
