import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime
import config
from model import CryClassifier, count_parameters
from dataset import get_dataloaders

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} Train', ncols=100, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        if hasattr(config, 'GRAD_CLIP'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, cm

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', ncols=100, leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC-AUC for binary classification
    if config.NUM_CLASSES == 2:
        all_probs = np.array(all_probs)[:, 1]
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
    else:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, auc, cm

def train_model():
    # Setup logging
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.MODEL_DIR, f'training_log_{timestamp}.txt')
    sys.stdout = Logger(log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'='*60)
    print(f'Training Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'='*60)
    print(f'Using device: {device}')
    
    # Load data
    train_loader, val_loader, test_loader, stats = get_dataloaders()
    
    # Check class distribution
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f'\nClass distribution:')
    for cls, count in zip(unique, counts):
        print(f'  Class {cls} ({config.CLASS_NAMES[cls]}): {count} samples ({count/len(train_labels)*100:.1f}%)')
    
    # Calculate class weights if enabled
    if hasattr(config, 'USE_CLASS_WEIGHTS') and config.USE_CLASS_WEIGHTS:
        class_weights = torch.FloatTensor([len(train_labels) / (len(unique) * c) for c in counts]).to(device)
        print(f'Class weights: {class_weights.cpu().numpy()}')
    else:
        class_weights = None
    
    # Save normalization stats for inference
    np.save(os.path.join(config.MODEL_DIR, 'stats.npy'), stats)
    print(f'\nSaved normalization stats - Mean: {stats["mean"]:.4f}, Std: {stats["std"]:.4f}')
    
    # Initialize model
    model = CryClassifier().to(device)
    params = count_parameters(model)
    print(f'\nModel parameters: {params:,}')
    if params > config.MAX_PARAMS:
        print(f'WARNING: Model has {params} parameters, exceeds limit of {config.MAX_PARAMS}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    if hasattr(config, 'USE_COSINE_SCHEDULE') and config.USE_COSINE_SCHEDULE:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        print('Using CosineAnnealingWarmRestarts scheduler')
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        print('Using ReduceLROnPlateau scheduler')
    
    # Training loop
    best_f1 = 0
    best_acc = 0
    patience_counter = 0
    
    print(f'\n{"="*60}')
    print('Starting Training Loop')
    print(f'{"="*60}')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc, train_f1, train_cm = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if hasattr(config, 'USE_COSINE_SCHEDULE') and config.USE_COSINE_SCHEDULE:
            scheduler.step()
        else:
            scheduler.step(val_f1)
        
        # Print metrics
        print(f'\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}')
        print(f'Train Confusion Matrix:\n{train_cm}')
        print(f'\nVal   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
        print(f'Val Confusion Matrix:\n{val_cm}')
        print(f'LR: {get_lr(optimizer):.6f}')
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'stats': stats
            }, os.path.join(config.MODEL_DIR, 'best_model.pth'))
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
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'), 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_auc, test_cm = validate(
        model, test_loader, criterion, device
    )
    
    print(f'\nFinal Test Results:')
    print(f'  Loss:     {test_loss:.4f}')
    print(f'  Accuracy: {test_acc:.4f}')
    print(f'  F1 Score: {test_f1:.4f}')
    print(f'  AUC:      {test_auc:.4f}')
    print(f'\nTest Confusion Matrix:')
    print(test_cm)
    print(f'\n{"="*60}')
    print(f'Training Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best Val F1: {best_f1:.4f}, Best Val Acc: {best_acc:.4f}')
    print(f'Log saved to: {log_file}')
    print(f'{"="*60}')
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

if __name__ == '__main__':
    train_model()