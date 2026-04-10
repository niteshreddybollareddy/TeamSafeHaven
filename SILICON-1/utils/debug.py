"""
Quick test to verify the model initialization fix
Run this before training to ensure the model works properly
"""

import torch
import numpy as np
from model import CryClassifier, count_parameters
import config

def test_1_different_outputs():
    """Test 1: Different inputs should produce different outputs"""
    print("="*60)
    print("TEST 1: Different Inputs → Different Outputs")
    print("="*60)
    
    model = CryClassifier()
    model.eval()
    
    # Create two different inputs
    torch.manual_seed(42)
    x1 = torch.randn(1, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    torch.manual_seed(123)
    x2 = torch.randn(1, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    
    diff = (out1 - out2).abs().max().item()
    
    print(f"Output 1: {out1[0].numpy()}")
    print(f"Output 2: {out2[0].numpy()}")
    print(f"Max difference: {diff:.6f}")
    
    if diff < 1e-5:
        print("❌ FAILED: Outputs are identical!")
        return False
    else:
        print("✓ PASSED: Outputs are different")
        return True

def test_2_weight_initialization():
    """Test 2: Weights should be properly initialized (not zeros/ones)"""
    print("\n" + "="*60)
    print("TEST 2: Weight Initialization")
    print("="*60)
    
    model = CryClassifier()
    
    all_zeros = True
    all_same = True
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            # Check if all zeros
            if param.abs().sum() > 0:
                all_zeros = False
            
            # Check if all same value
            if param.std() > 1e-6:
                all_same = False
            
            print(f"{name:40s} Mean: {param.mean().item():8.4f}, Std: {param.std().item():8.4f}")
    
    if all_zeros:
        print("\n❌ FAILED: All weights are zero!")
        return False
    elif all_same:
        print("\n❌ FAILED: All weights have same value!")
        return False
    else:
        print("\n✓ PASSED: Weights are properly initialized")
        return True

def test_3_gradient_flow():
    """Test 3: Gradients should flow through the network"""
    print("\n" + "="*60)
    print("TEST 3: Gradient Flow")
    print("="*60)
    
    model = CryClassifier()
    model.train()
    
    x = torch.randn(2, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    labels = torch.tensor([0, 1])
    
    criterion = torch.nn.CrossEntropyLoss()
    
    output = model(x)
    loss = criterion(output, labels)
    loss.backward()
    
    # Check if gradients exist
    has_grad = False
    zero_grad = False
    grad_stats = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            
            if grad_mean > 0:
                has_grad = True
            if grad_max == 0:
                zero_grad = True
            
            grad_stats.append((name, grad_mean, grad_max))
    
    # Print gradient statistics
    print("Gradient statistics (top 5 layers):")
    for name, mean, max_val in sorted(grad_stats, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name:40s} Mean: {mean:.6f}, Max: {max_val:.6f}")
    
    if not has_grad:
        print("\n❌ FAILED: No gradients detected!")
        return False
    elif zero_grad:
        print("\n⚠️  WARNING: Some layers have zero gradients")
        return True
    else:
        print("\n✓ PASSED: Gradients are flowing")
        return True

def test_4_overfit_single_batch():
    """Test 4: Model should be able to overfit on a single batch"""
    print("\n" + "="*60)
    print("TEST 4: Overfitting Single Batch")
    print("="*60)
    
    model = CryClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create a small batch with clear labels
    x = torch.randn(4, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    labels = torch.tensor([0, 1, 0, 1])
    
    print("\nTraining for 100 steps on 4 samples...")
    print(f"Target labels: {labels.tolist()}")
    
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            preds = output.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            print(f"  Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:.2f}, Preds={preds.tolist()}")
    
    # Final accuracy
    model.eval()
    with torch.no_grad():
        output = model(x)
        preds = output.argmax(dim=1)
        final_acc = (preds == labels).float().mean().item()
    
    print(f"\nFinal accuracy: {final_acc:.2f}")
    print(f"Final predictions: {preds.tolist()}")
    
    if final_acc < 0.75:
        print("❌ FAILED: Model cannot overfit on 4 samples!")
        print("   This indicates a serious problem with the model.")
        return False
    else:
        print(f"✓ PASSED: Model achieved {final_acc*100:.0f}% accuracy")
        return True

def test_5_model_stats():
    """Test 5: Check model statistics"""
    print("\n" + "="*60)
    print("TEST 5: Model Statistics")
    print("="*60)
    
    model = CryClassifier()
    total_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter limit: {config.MAX_PARAMS:,}")
    
    if total_params > config.MAX_PARAMS:
        print(f"⚠️  WARNING: Model exceeds parameter limit by {total_params - config.MAX_PARAMS:,}")
    else:
        print(f"✓ Model within parameter limit ({config.MAX_PARAMS - total_params:,} parameters remaining)")
    
    # Test with expected input size
    x = torch.randn(1, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected output shape: (1, {config.NUM_CLASSES})")
        
        if output.shape[1] != config.NUM_CLASSES:
            print(f"❌ FAILED: Wrong output dimension!")
            return False
        
        return True
    except Exception as e:
        print(f"\n❌ FAILED: Error in forward pass: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "MODEL INITIALIZATION TEST")
    print("="*70)
    
    results = {
        "Different outputs": test_1_different_outputs(),
        "Weight initialization": test_2_weight_initialization(),
        "Gradient flow": test_3_gradient_flow(),
        "Overfit single batch": test_4_overfit_single_batch(),
        "Model statistics": test_5_model_stats()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Model is ready for training.")
        print("   Run: python train.py")
    else:
        print("⚠️  SOME TESTS FAILED. Fix the issues before training.")
    print("="*70)
    
    return all_passed

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)