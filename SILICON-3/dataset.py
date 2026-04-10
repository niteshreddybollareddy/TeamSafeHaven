import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import config

class CryDataset(Dataset):
    def __init__(self, file_paths, labels, stats, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = stats['mean']
        self.std = stats['std']
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def time_mask(self, mel_spec, num_masks=2, mask_param=40):
        """Apply time masking (SpecAugment)"""
        mel_spec = mel_spec.clone()
        _, _, time_frames = mel_spec.shape
        
        for _ in range(num_masks):
            t = np.random.randint(0, min(mask_param, time_frames))
            t0 = np.random.randint(0, time_frames - t)
            mel_spec[:, :, t0:t0+t] = 0
        
        return mel_spec
    
    def freq_mask(self, mel_spec, num_masks=2, mask_param=8):
        """Apply frequency masking (SpecAugment)"""
        mel_spec = mel_spec.clone()
        _, freq_bands, _ = mel_spec.shape
        
        for _ in range(num_masks):
            f = np.random.randint(0, min(mask_param, freq_bands))
            f0 = np.random.randint(0, freq_bands - f)
            mel_spec[:, f0:f0+f, :] = 0
        
        return mel_spec
    
    def add_noise(self, mel_spec, noise_factor=0.005):
        """Add random noise to mel-spectrogram"""
        noise = torch.randn_like(mel_spec) * noise_factor
        return mel_spec + noise
    
    def time_shift(self, mel_spec, shift_max=40):
        """Shift mel-spectrogram in time"""
        _, _, time_frames = mel_spec.shape
        shift = np.random.randint(-shift_max, shift_max)
        
        if shift > 0:
            mel_spec = torch.cat([mel_spec[:, :, shift:], mel_spec[:, :, :shift]], dim=2)
        elif shift < 0:
            mel_spec = torch.cat([mel_spec[:, :, shift:], mel_spec[:, :, :shift]], dim=2)
        
        return mel_spec
    
    def augment_data(self, mel_spec):
        """Apply random augmentations"""
        # Apply augmentations with probability
        if np.random.random() < 0.5:
            mel_spec = self.time_mask(mel_spec, num_masks=1, mask_param=40)
        
        if np.random.random() < 0.5:
            mel_spec = self.freq_mask(mel_spec, num_masks=1, mask_param=8)
        
        if np.random.random() < 0.3:
            mel_spec = self.add_noise(mel_spec, noise_factor=0.003)
        
        if np.random.random() < 0.3:
            mel_spec = self.time_shift(mel_spec, shift_max=30)
        
        return mel_spec
    
    def __getitem__(self, idx):
        mel_spec = np.load(self.file_paths[idx])
        
        # Normalize using training stats
        mel_spec = (mel_spec - self.mean) / (self.std + 1e-8)
        
        # Add channel dimension: (F, T) -> (1, F, T)
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        # Apply augmentation if training
        if self.augment:
            mel_spec = self.augment_data(mel_spec)
        
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return mel_spec, label

def load_data():
    """Load processed data and create train/val/test splits"""
    file_paths = []
    labels = []
    
    print("\nLoading dataset...")
    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.PROCESSED_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"ERROR: Directory not found: {class_dir}")
            print("Please run preprocess.py first!")
            raise FileNotFoundError(f"Directory not found: {class_dir}")
        
        files = list(Path(class_dir).glob('*.npy'))
        print(f"  {class_name}: {len(files)} files")
        
        if len(files) == 0:
            print(f"WARNING: No .npy files found in {class_dir}")
        
        file_paths.extend([str(f) for f in files])
        labels.extend([idx] * len(files))
    
    if len(file_paths) == 0:
        raise ValueError("No data files found! Please run preprocess.py first.")
    
    print(f"\nTotal files: {len(file_paths)}")
    
    # Check class balance
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label} ({config.CLASS_NAMES[label]}): {count} ({count/len(labels)*100:.1f}%)")
    
    # Warn if highly imbalanced
    if max(counts) / min(counts) > 2:
        print("\nWARNING: Dataset is imbalanced! Consider using class weights.")
    
    # Split data with stratification
    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            file_paths, labels, 
            test_size=(1-config.TRAIN_SPLIT), 
            random_state=42, 
            stratify=labels
        )
        
        val_size = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, 
            test_size=(1-val_size), 
            random_state=42, 
            stratify=temp_labels
        )
    except ValueError as e:
        print(f"\nERROR during data splitting: {e}")
        print("This might happen if you have too few samples per class.")
        raise
    
    # Calculate normalization stats from training data only
    print("\nCalculating normalization statistics from training data...")
    train_values = []
    for f in train_files[:100]:  # Sample for speed
        train_values.append(np.load(f).flatten())
    train_values = np.concatenate(train_values)
    
    stats = {
        'mean': float(train_values.mean()), 
        'std': float(train_values.std())
    }
    
    print(f'Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')
    print(f'Normalization stats - Mean: {stats["mean"]:.4f}, Std: {stats["std"]:.4f}')
    
    # Check if std is too small (might cause issues)
    if stats['std'] < 1.0:
        print(f"WARNING: Standard deviation is very small ({stats['std']:.4f})")
    
    # Create datasets with augmentation only for training
    use_augment = hasattr(config, 'USE_AUGMENTATION') and config.USE_AUGMENTATION
    
    train_dataset = CryDataset(train_files, train_labels, stats, augment=use_augment)
    val_dataset = CryDataset(val_files, val_labels, stats, augment=False)
    test_dataset = CryDataset(test_files, test_labels, stats, augment=False)
    
    if use_augment:
        print("Data augmentation enabled for training")
    
    return train_dataset, val_dataset, test_dataset, stats

def get_dataloaders(batch_size=None):
    """Get dataloaders for training, validation, and testing"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_dataset, val_dataset, test_dataset, stats = load_data()
    
    # Use pin_memory for faster GPU transfer
    pin_memory = torch.cuda.is_available()
    
    # Use appropriate number of workers
    num_workers = 4 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, stats

if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader, stats = get_dataloaders()
    
    print("\nTesting data loading...")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels[:5].tolist()}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
        
        if batch_idx == 2:
            break
    
    print("\nDataset loading test completed successfully!")