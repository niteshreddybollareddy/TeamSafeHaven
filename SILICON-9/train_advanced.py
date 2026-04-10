"""
SILICON-9: Advanced Training Script with Experiment Tracking
Production-grade training pipeline for CryNet-Micro-v2
"""

import os
import sys
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from dataset import get_dataloaders, load_data
import config
from model import CryNetMicroV2, count_parameters


class ExperimentTracker:
    """Track experiments with detailed metrics and configurations"""

    def __init__(self, experiment_name=None):
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.experiment_dir = Path('experiments') / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.logs_dir = self.experiment_dir / 'logs'
        self.graphs_dir = self.experiment_dir / 'graphs'

        for d in [self.checkpoints_dir, self.logs_dir, self.graphs_dir]:
            d.mkdir(exist_ok=True)

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'train_precision': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'val_auc': [],
            'lr': [], 'epoch_time': []
        }

        self.best_metrics = {
            'best_val_f1': 0,
            'best_val_acc': 0,
            'best_val_auc': 0,
            'best_epoch': 0
        }

        # Save configuration
        self.save_config()

    def save_config(self):
        """Save experiment configuration"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model': {
                'architecture': 'CryNetMicroV2',
                'num_classes': config.NUM_CLASSES,
                'cnn_filters': config.CNN_FILTERS,
                'dropout': config.DROPOUT,
                'max_params': config.MAX_PARAMS
            },
            'training': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'num_epochs': config.NUM_EPOCHS,
                'patience': config.PATIENCE,
                'weight_decay': config.WEIGHT_DECAY,
                'label_smoothing': config.LABEL_SMOOTHING
            },
            'data': {
                'sample_rate_hz': config.SAMPLE_RATE_HZ,
                'sample_length_ms': config.SAMPLE_LENGTH_MS,
                'sample_length': config.SAMPLE_LENGTH,
                'n_mels': config.FILTERBANK_N_CHANNELS,
                'n_fft': config.FFT_LENGTH,
                'window_size_ms': config.WINDOW_SIZE_MS,
                'window_step_ms': config.WINDOW_STEP_MS,
                'fmin': config.FILTERBANK_LOWER_BAND_LIMIT,
                'fmax': config.FILTERBANK_UPPER_BAND_LIMIT,
                'spectrogram_shape': config.SPECTROGRAM_SHAPE,
                'n_time_frames': config.N_TIME_FRAMES
            },
            'augmentation': {
                'use_augmentation': config.USE_AUGMENTATION,
                'use_spec_augment': config.USE_SPEC_AUGMENT,
                'freq_mask_param': config.FREQ_MASK_PARAM,
                'time_mask_param': config.TIME_MASK_PARAM,
                'num_freq_masks': config.NUM_FREQ_MASKS,
                'num_time_masks': config.NUM_TIME_MASKS,
                'use_mixup': config.USE_MIXUP,
                'mixup_alpha': config.MIXUP_ALPHA
            }
        }

        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

    def log_epoch(self, epoch, metrics, epoch_time):
        """Log metrics for an epoch"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))

        self.history['epoch_time'].append(float(epoch_time))

        # Update best metrics
        if metrics['val_f1'] > self.best_metrics['best_val_f1']:
            self.best_metrics['best_val_f1'] = float(metrics['val_f1'])
            self.best_metrics['best_val_acc'] = float(metrics['val_acc'])
            self.best_metrics['best_val_auc'] = float(metrics['val_auc'])
            self.best_metrics['best_epoch'] = epoch

        # Save history after each epoch
        self.save_history()

    def save_history(self):
        """Save training history"""
        with open(self.logs_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        with open(self.logs_dir / 'best_metrics.json', 'w') as f:
            json.dump(self.best_metrics, f, indent=2)

    def save_test_results(self, test_metrics):
        """Save final test results"""
        with open(self.logs_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)

    def plot_training_curves(self):
        """Generate comprehensive training visualizations"""

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = {'train': '#e74c3c', 'val': '#3498db'}

        # 1. Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)

        ax.plot(epochs, self.history['train_loss'], label='Train Loss',
                color=colors['train'], linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, self.history['val_loss'], label='Val Loss',
                color=colors['val'], linewidth=2, marker='s', markersize=4)

        ax.axvline(self.best_metrics['best_epoch'] + 1, color='green',
                   linestyle='--', linewidth=2, alpha=0.7, label='Best Model')

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graphs_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy curves
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, self.history['train_acc'], label='Train Accuracy',
                color=colors['train'], linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, self.history['val_acc'], label='Val Accuracy',
                color=colors['val'], linewidth=2, marker='s', markersize=4)

        ax.axvline(self.best_metrics['best_epoch'] + 1, color='green',
                   linestyle='--', linewidth=2, alpha=0.7, label='Best Model')

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Training and Validation Accuracy', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graphs_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. F1 Score curves
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, self.history['train_f1'], label='Train F1',
                color=colors['train'], linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, self.history['val_f1'], label='Val F1',
                color=colors['val'], linewidth=2, marker='s', markersize=4)

        ax.axvline(self.best_metrics['best_epoch'] + 1, color='green',
                   linestyle='--', linewidth=2, alpha=0.7, label='Best Model')

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
        ax.set_title('Training and Validation F1 Score', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graphs_dir / 'f1_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Multi-metric comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy
        axes[0, 0].plot(epochs, self.history['val_acc'], color=colors['val'], linewidth=2)
        axes[0, 0].set_title('Validation Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)

        # F1 Score
        axes[0, 1].plot(epochs, self.history['val_f1'], color=colors['val'], linewidth=2)
        axes[0, 1].set_title('Validation F1 Score', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].grid(True, alpha=0.3)

        # AUC
        axes[1, 0].plot(epochs, self.history['val_auc'], color=colors['val'], linewidth=2)
        axes[1, 0].set_title('Validation AUC', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs, self.history['lr'], color='#e67e22', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graphs_dir / 'multi_metric_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, cm, class_names, split='test'):
        """Plot confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 7))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion'}, ax=ax,
                   square=True, linewidths=1, linecolor='gray')

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {split.capitalize()} Set',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.graphs_dir / f'confusion_matrix_{split}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.plot(fpr, tpr, color='#3498db', linewidth=3,
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random')

        ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - Binary Classification', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graphs_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curve(self, precision, recall, pr_auc):
        """Plot Precision-Recall curve"""
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.plot(recall, precision, color='#2ecc71', linewidth=3,
               label=f'PR curve (AUC = {pr_auc:.4f})')

        ax.fill_between(recall, precision, alpha=0.2, color='#2ecc71')

        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.graphs_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_model_architecture(self, model):
        """Save model architecture diagram and summary"""
        # Save model summary to text file
        with open(self.logs_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Save model architecture as image
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=self.graphs_dir / 'model_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',  # Top to Bottom
                expand_nested=True,
                dpi=200,
                show_layer_activations=True
            )
            print(f"✓ Model architecture saved to {self.graphs_dir / 'model_architecture.png'}")
        except Exception as e:
            print(f"⚠ Could not save model architecture diagram: {e}")
            print("  (Install graphviz: brew install graphviz or apt-get install graphviz)")

        # Create layer-wise parameter count visualization
        layer_params = []
        layer_names = []
        for layer in model.layers:
            params = layer.count_params()
            if params > 0:
                layer_params.append(params)
                layer_names.append(layer.name)

        if layer_params:
            fig, ax = plt.subplots(figsize=(10, max(6, len(layer_names) * 0.4)))
            colors = plt.cm.viridis(np.linspace(0, 1, len(layer_params)))

            bars = ax.barh(range(len(layer_params)), layer_params, color=colors)
            ax.set_yticks(range(len(layer_params)))
            ax.set_yticklabels(layer_names, fontsize=9)
            ax.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')
            ax.set_title(f'Layer-wise Parameter Distribution\nTotal: {sum(layer_params):,} parameters',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, layer_params)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f' {val:,}', ha='left', va='center', fontsize=8)

            plt.tight_layout()
            plt.savefig(self.graphs_dir / 'parameter_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Parameter distribution saved to {self.graphs_dir / 'parameter_distribution.png'}")


class AdvancedTrainer:
    """Advanced trainer with detailed logging and experiment tracking"""

    def __init__(self, experiment_name=None):
        self.tracker = ExperimentTracker(experiment_name)
        self.device = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'

    def get_lr(self, optimizer):
        """Get current learning rate"""
        if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            return optimizer.learning_rate(optimizer.iterations).numpy()
        return float(optimizer.learning_rate.numpy())

    @tf.function
    def train_step(self, model, inputs, labels, criterion, optimizer):
        """Single training step"""
        # Convert integer labels to one-hot encoding for categorical crossentropy
        labels_onehot = tf.one_hot(labels, depth=config.NUM_CLASSES)

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = criterion(labels_onehot, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, outputs

    def train_epoch(self, model, loader, criterion, optimizer, epoch):
        """Train for one epoch with detailed metrics"""
        import time
        start_time = time.time()

        total_loss = 0.0
        all_preds, all_labels = [], []
        num_batches = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', ncols=120, leave=False)
        for inputs, labels in pbar:
            loss, outputs = self.train_step(model, inputs, labels, criterion, optimizer)

            total_loss += loss.numpy()
            num_batches += 1

            preds = tf.argmax(outputs, axis=1).numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            pbar.set_postfix({
                'loss': f'{loss.numpy():.4f}',
                'lr': f'{self.get_lr(optimizer):.6f}'
            })

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        epoch_time = time.time() - start_time

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'cm': cm,
            'epoch_time': epoch_time
        }

    def validate(self, model, loader, criterion, return_probs=False):
        """Validate with detailed metrics"""
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        num_batches = 0

        pbar = tqdm(loader, desc='Validating', ncols=120, leave=False)
        for inputs, labels in pbar:
            # Convert integer labels to one-hot encoding
            labels_onehot = tf.one_hot(labels, depth=config.NUM_CLASSES)

            outputs = model(inputs, training=False)
            loss = criterion(labels_onehot, outputs)

            total_loss += loss.numpy()
            num_batches += 1

            probs = outputs.numpy()
            preds = tf.argmax(outputs, axis=1).numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

        # ROC-AUC
        all_probs_arr = np.array(all_probs)[:, 1]
        try:
            auc_score = roc_auc_score(all_labels, all_probs_arr)
        except:
            auc_score = 0.0

        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc_score,
            'cm': cm
        }

        if return_probs:
            metrics['labels'] = all_labels
            metrics['probs'] = all_probs

        return metrics

    def train(self):
        """Main training loop"""

        print(f"\n{'='*80}")
        print(f"SILICON-9: ADVANCED TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Experiment: {self.tracker.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Load data
        print("Loading dataset...")
        train_loader, val_loader, test_loader, stats = get_dataloaders()
        train_gen, _, _, _ = load_data()
        train_size = len(train_gen)

        print(f"✓ Dataset loaded")
        print(f"  Train: {train_size:,} samples")
        print(f"  Normalization: Mean={stats['mean']:.6f}, Std={stats['std']:.6f}")

        # Save stats
        np.save(self.tracker.checkpoints_dir / 'stats.npy', stats)

        # Class distribution
        train_labels = train_gen.labels
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f"\n  Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"    {config.CLASS_NAMES[cls]:12s}: {count:,} ({count/len(train_labels)*100:.1f}%)")

        # Initialize model
        print(f"\nInitializing model...")
        model = CryNetMicroV2(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)

        dummy_input = tf.zeros((1, 1, 64, 98))
        model(dummy_input)

        total_params = count_parameters(model)
        print(f"✓ Model: CryNet-Micro-v2")
        print(f"  Parameters: {total_params:,} / {config.MAX_PARAMS:,}")
        print(f"  Status: {'✓ Within budget' if total_params <= config.MAX_PARAMS else '❌ Over budget'}")

        # Save model architecture and summary
        print(f"\n💾 Saving model architecture visualization...")
        self.tracker.save_model_architecture(model)

        # Loss and optimizer
        criterion = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=config.LABEL_SMOOTHING
        )

        steps_per_epoch = train_size // config.BATCH_SIZE
        total_steps = config.NUM_EPOCHS * steps_per_epoch

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=total_steps,
            alpha=0.0
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.WEIGHT_DECAY
        )

        print(f"\n✓ Training configuration:")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Epochs: {config.NUM_EPOCHS}")
        print(f"  Early stopping patience: {config.PATIENCE}")

        # Training loop
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}\n")

        best_f1 = 0
        patience_counter = 0

        for epoch in range(config.NUM_EPOCHS):
            print(f"\n{'─'*80}")
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            print(f"{'─'*80}")

            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch+1)

            # Validate
            val_metrics = self.validate(model, val_loader, criterion)

            # Log to tracker
            self.tracker.log_epoch(epoch, {
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_auc': val_metrics['auc'],
                'lr': self.get_lr(optimizer)
            }, train_metrics['epoch_time'])

            # Print metrics
            print(f"\n📊 Train Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f} | P: {train_metrics['precision']:.4f} | "
                  f"R: {train_metrics['recall']:.4f}")

            print(f"\n📈 Val Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

            print(f"\n⏱  Time: {train_metrics['epoch_time']:.2f}s | LR: {self.get_lr(optimizer):.6f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0

                model.save_weights(self.tracker.checkpoints_dir / 'best_model.weights.h5')

                meta_data = {
                    'epoch': epoch + 1,
                    'val_f1': float(val_metrics['f1']),
                    'val_acc': float(val_metrics['accuracy']),
                    'val_auc': float(val_metrics['auc']),
                    'total_params': int(total_params),
                    'stats': stats
                }
                np.save(self.tracker.checkpoints_dir / 'meta_data.npy', meta_data)

                print(f"\n✓ New best model saved (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"\n⏳ Patience: {patience_counter}/{config.PATIENCE}")

            # Early stopping
            if patience_counter >= config.PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                break

        # Test evaluation
        print(f"\n{'='*80}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*80}\n")

        model.load_weights(self.tracker.checkpoints_dir / 'best_model.weights.h5')
        test_metrics = self.validate(model, test_loader, criterion, return_probs=True)

        print(f"📊 Test Results:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  AUC:       {test_metrics['auc']:.4f}")

        # Per-class metrics
        print(f"\n📋 Per-Class Metrics:")
        cm = test_metrics['cm']
        for i in range(config.NUM_CLASSES):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            print(f"  {config.CLASS_NAMES[i]}:")
            print(f"    Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        # Save test results
        test_results = {
            'accuracy': float(test_metrics['accuracy']),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'auc': float(test_metrics['auc']),
            'confusion_matrix': test_metrics['cm'].tolist()
        }
        self.tracker.save_test_results(test_results)

        # Generate visualizations
        print(f"\n📊 Generating visualizations...")
        self.tracker.plot_training_curves()
        self.tracker.plot_confusion_matrix(test_metrics['cm'], config.CLASS_NAMES, 'test')

        # ROC and PR curves
        test_probs = np.array(test_metrics['probs'])[:, 1]
        test_labels = test_metrics['labels']

        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        roc_auc = auc(fpr, tpr)
        self.tracker.plot_roc_curve(fpr, tpr, roc_auc)

        precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_probs)
        pr_auc = auc(recall_curve, precision_curve)
        self.tracker.plot_pr_curve(precision_curve, recall_curve, pr_auc)

        print(f"✓ Visualizations saved to: {self.tracker.graphs_dir}")

        # Final summary
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Experiment: {self.tracker.experiment_name}")
        print(f"Best Val F1: {self.tracker.best_metrics['best_val_f1']:.4f} (Epoch {self.tracker.best_metrics['best_epoch']+1})")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"\nResults saved to: {self.tracker.experiment_dir}")
        print(f"{'='*80}\n")


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create experiment name with timestamp
    experiment_name = f"crynet_micro_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize and run trainer
    trainer = AdvancedTrainer(experiment_name=experiment_name)
    trainer.train()


if __name__ == '__main__':
    main()
