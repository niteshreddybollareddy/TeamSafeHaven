"""
Evaluate the best saved model on test set
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import config
from model import CryClassifier
from dataset import get_dataloaders

def evaluate_model():
    print("="*60)
    print("Evaluating Best Model on Test Set")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, _, test_loader, stats = get_dataloaders()
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print("\nLoading best model...")
    model = CryClassifier().to(device)
    
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    # Fix for PyTorch 2.6+ - add weights_only=False
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Validation F1: {checkpoint['val_f1']:.4f}")
    print(f"  Validation Acc: {checkpoint['val_acc']:.4f}")
    print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Running Evaluation on Test Set")
    print("="*60)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC-AUC
    if config.NUM_CLASSES == 2:
        all_probs_class1 = np.array(all_probs)[:, 1]
        test_auc = roc_auc_score(all_labels, all_probs_class1)
    else:
        test_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC-ROC:  {test_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Cry   Not Cry")
    print(f"Actual Cry       {cm[0][0]:3d}     {cm[0][1]:3d}")
    print(f"       Not Cry   {cm[1][0]:3d}     {cm[1][1]:3d}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"  Cry class:")
    print(f"    Precision: {cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0:.4f}")
    print(f"    Recall:    {cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0:.4f}")
    
    print(f"  Not Cry class:")
    print(f"    Precision: {cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0:.4f}")
    print(f"    Recall:    {cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=config.CLASS_NAMES,
                               digits=4))
    
    # Error analysis
    print(f"\nError Analysis:")
    false_positives = cm[1][0]  # Not cry predicted as cry
    false_negatives = cm[0][1]  # Cry predicted as not cry
    print(f"  False Positives (Not Cry → Cry): {false_positives}")
    print(f"  False Negatives (Cry → Not Cry): {false_negatives}")
    print(f"  Total Errors: {false_positives + false_negatives} out of {len(all_labels)} samples")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    # Summary
    if test_acc >= 0.95:
        print("\n🎉 EXCELLENT! Model achieved >95% accuracy!")
    elif test_acc >= 0.90:
        print("\n✓ GREAT! Model achieved >90% accuracy!")
    elif test_acc >= 0.85:
        print("\n✓ GOOD! Model achieved >85% accuracy!")
    else:
        print("\n⚠️  Model accuracy below 85%. Consider more training or data augmentation.")
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
    }

if __name__ == '__main__':
    results = evaluate_model()