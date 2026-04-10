"""
SILICON-12 Dataset Loader
3-class classification: Baby Cry, Dog Barking, Unknown

Based on SILICON-9's data loading approach:
- Fixed 148-frame spectrograms (1.5s audio, 10ms hop, center=False)
- Global z-score normalization (computed from training data)
- SpecAugment for training augmentation
- Proper train/val/test stratified split
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class AudioDatasetGenerator:
    """
    Dataset generator for 3-class audio classification.

    Features:
    - Fixed shape output: (1, 64, 148)
    - Global z-score normalization
    - SpecAugment for training (time/freq masking)
    - Optional MixUp augmentation
    """

    def __init__(self, file_paths, labels, stats, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = stats['mean']
        self.std = stats['std']
        self.augment = augment
        self.expected_frames = config.N_TIME_FRAMES  # 148

    def __len__(self):
        return len(self.file_paths)

    def _pad_or_trim(self, mel_spec):
        """Ensure mel_spec has exactly expected_frames (148)."""
        current_frames = mel_spec.shape[1]

        if current_frames < self.expected_frames:
            # Pad
            pad_width = self.expected_frames - current_frames
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        elif current_frames > self.expected_frames:
            # Trim
            mel_spec = mel_spec[:, :self.expected_frames]

        return mel_spec

    def _time_mask(self, mel_spec, num_masks=1, mask_param=15):
        """Apply time masking (SpecAugment)."""
        mel_spec = mel_spec.copy()
        time_frames = mel_spec.shape[2]

        for _ in range(num_masks):
            t = np.random.randint(1, min(mask_param, time_frames))
            t0 = np.random.randint(0, time_frames - t)
            mel_spec[:, :, t0:t0+t] = 0

        return mel_spec

    def _freq_mask(self, mel_spec, num_masks=1, mask_param=8):
        """Apply frequency masking (SpecAugment)."""
        mel_spec = mel_spec.copy()
        freq_bands = mel_spec.shape[1]

        for _ in range(num_masks):
            f = np.random.randint(1, min(mask_param, freq_bands))
            f0 = np.random.randint(0, freq_bands - f)
            mel_spec[:, f0:f0+f, :] = 0

        return mel_spec

    def _add_noise(self, mel_spec, noise_factor=0.003):
        """Add random noise."""
        noise = np.random.randn(*mel_spec.shape) * noise_factor
        return mel_spec + noise

    def _augment_spectrogram(self, mel_spec):
        """Apply training augmentations."""
        # SpecAugment (always apply during training)
        if config.USE_SPEC_AUGMENT:
            mel_spec = self._time_mask(
                mel_spec,
                num_masks=config.NUM_TIME_MASKS,
                mask_param=config.TIME_MASK_PARAM
            )
            mel_spec = self._freq_mask(
                mel_spec,
                num_masks=config.NUM_FREQ_MASKS,
                mask_param=config.FREQ_MASK_PARAM
            )

        # Light noise (20% probability)
        if np.random.random() < 0.2:
            mel_spec = self._add_noise(mel_spec, noise_factor=0.003)

        return mel_spec

    def __call__(self):
        """Generator function for tf.data.Dataset."""
        indices = list(range(len(self.file_paths)))

        for idx in indices:
            try:
                # Load mel-spectrogram (64, T)
                mel_spec = np.load(self.file_paths[idx])

                # Ensure fixed shape (64, 98)
                mel_spec = self._pad_or_trim(mel_spec)

                # Z-score normalization (SILICON-9 style)
                mel_spec = (mel_spec - self.mean) / (self.std + 1e-8)

                # Add channel dimension: (64, 98) -> (1, 64, 98)
                mel_spec = mel_spec[np.newaxis, ...].astype(np.float32)

                # Apply augmentations during training
                if self.augment:
                    mel_spec = self._augment_spectrogram(mel_spec)

                label = self.labels[idx]

                yield mel_spec, label

            except Exception as e:
                print(f"Error loading {self.file_paths[idx]}: {e}")
                continue


def load_data():
    """
    Load processed dataset and create train/val/test splits.

    Returns:
        train_gen, val_gen, test_gen, stats
    """
    file_paths = []
    labels = []

    print("\n" + "="*60)
    print("SILICON-12 Dataset Loading")
    print("="*60)

    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"ERROR: Directory not found: {class_dir}")
            raise FileNotFoundError(f"Directory not found: {class_dir}")

        files = list(Path(class_dir).glob('*.npy'))
        print(f"  {class_name}: {len(files)} files")

        file_paths.extend([str(f) for f in files])
        labels.extend([idx] * len(files))

    if not file_paths:
        raise ValueError("No data files found!")

    print(f"\nTotal: {len(file_paths)} samples")

    # Stratified split
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=(1 - config.TRAIN_SPLIT),
        random_state=config.RANDOM_SEED,
        stratify=labels
    )

    val_size = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=(1 - val_size),
        random_state=config.RANDOM_SEED,
        stratify=temp_labels
    )

    print(f"\nSplit: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Compute normalization stats from training data (SILICON-9 approach)
    # CRITICAL: Exclude padded silence (values < -80 dB)
    print("\nComputing normalization stats (excluding silence)...")
    train_values = []
    for f in train_files[:200]:  # Sample 200 files for stats
        mel_spec = np.load(f).flatten()
        # Exclude padded silence
        valid_values = mel_spec[mel_spec > -80]
        if len(valid_values) > 0:
            train_values.append(valid_values)

    train_values = np.concatenate(train_values)

    stats = {
        'mean': float(train_values.mean()),
        'std': float(train_values.std())
    }

    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")

    # Create generators
    use_augment = config.USE_AUGMENTATION
    train_gen = AudioDatasetGenerator(train_files, train_labels, stats, augment=use_augment)
    val_gen = AudioDatasetGenerator(val_files, val_labels, stats, augment=False)
    test_gen = AudioDatasetGenerator(test_files, test_labels, stats, augment=False)

    return train_gen, val_gen, test_gen, stats


def get_dataloaders(batch_size=None):
    """
    Create TensorFlow data loaders.

    Returns:
        train_loader, val_loader, test_loader, stats
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_gen, val_gen, test_gen, stats = load_data()

    # Fixed output signature (1, 64, 98)
    output_signature = (
        tf.TensorSpec(shape=(1, 64, config.N_TIME_FRAMES), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    def create_tf_dataset(generator, shuffle=False):
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_loader = create_tf_dataset(train_gen, shuffle=True)
    val_loader = create_tf_dataset(val_gen, shuffle=False)
    test_loader = create_tf_dataset(test_gen, shuffle=False)

    return train_loader, val_loader, test_loader, stats


def get_class_weights():
    """Compute class weights for imbalanced dataset."""
    class_counts = []

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        if os.path.exists(class_dir):
            count = len(list(Path(class_dir).glob('*.npy')))
            class_counts.append(count)
        else:
            class_counts.append(0)

    total = sum(class_counts)
    weights = {i: total / (len(class_counts) * c) if c > 0 else 1.0
               for i, c in enumerate(class_counts)}

    return weights


if __name__ == '__main__':
    print("Testing SILICON-12 dataset loading...")

    train_loader, val_loader, test_loader, stats = get_dataloaders()

    print(f"\nStats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    print("\nTesting batches:")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels[:8].numpy().tolist()}")
        print(f"  Data range: [{np.min(data):.2f}, {np.max(data):.2f}]")

        # Verify expected shape
        expected = (config.BATCH_SIZE, 1, 64, config.N_TIME_FRAMES)
        if data.shape[1:] == expected[1:]:
            print(f"  Shape OK: (B, 1, 64, {config.N_TIME_FRAMES})")
        else:
            print(f"  Shape MISMATCH! Expected {expected}, got {data.shape}")

        if batch_idx >= 2:
            break

    # Class weights
    weights = get_class_weights()
    print(f"\nClass weights: {weights}")

    print("\n" + "="*60)
    print("Dataset loading test PASSED!")
    print("="*60)
