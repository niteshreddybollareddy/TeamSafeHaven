import torch
import torch.nn as nn
import torch.nn.init as init
import config

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Proper initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class CryClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        
        # Initial conv with BatchNorm - reduced channels
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x437 -> 32x218
        
        # Depthwise separable blocks with pooling - optimized channel sizes
        self.ds_conv1 = DepthwiseSeparableConv(12, 24, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x218 -> 16x109
        
        self.ds_conv2 = DepthwiseSeparableConv(24, 48, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x109 -> 8x54
        
        self.ds_conv3 = DepthwiseSeparableConv(48, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 8x54 -> 4x27
        
        self.ds_conv4 = DepthwiseSeparableConv(64, 96, kernel_size=3, stride=1, padding=1)
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(96, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Kaiming/He initialization for ReLU networks
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # DS conv blocks
        x = self.ds_conv1(x)
        x = self.pool2(x)
        
        x = self.ds_conv2(x)
        x = self.pool3(x)
        
        x = self.ds_conv3(x)
        x = self.pool4(x)
        
        x = self.ds_conv4(x)
        
        # Classifier
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_initialization():
    """Test that model produces different outputs for different inputs"""
    model = CryClassifier()
    model.eval()
    
    # Create two different random inputs
    x1 = torch.randn(1, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    x2 = torch.randn(1, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    
    # Check if outputs are different
    if torch.allclose(out1, out2, atol=1e-5):
        print("❌ ERROR: Model outputs are identical!")
        return False
    else:
        print("✓ Model produces different outputs for different inputs")
        print(f"  Output 1: {out1.numpy()}")
        print(f"  Output 2: {out2.numpy()}")
        print(f"  Difference: {(out1 - out2).abs().max().item():.6f}")
        return True

if __name__ == '__main__':
    model = CryClassifier()
    params = count_parameters(model)
    print(f'Total parameters: {params:,}')
    print(f'Parameter budget: {config.MAX_PARAMS:,}')
    
    if params <= config.MAX_PARAMS:
        print(f'✓ Within budget! ({config.MAX_PARAMS - params:,} parameters remaining)')
    else:
        print(f'❌ Exceeds budget by {params - config.MAX_PARAMS:,} parameters')
    
    # Test forward pass
    x = torch.randn(2, 1, config.N_MELS, config.MAX_TIME_FRAMES)
    out = model(x)
    print(f'\nInput shape: {x.shape}')
    print(f'Output shape: {out.shape}')
    
    print("\nTesting model initialization...")
    test_model_initialization()
    
    # Check that different batches produce different outputs
    print("\nTesting batch outputs...")
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"Batch output 0: {out[0].numpy()}")
        print(f"Batch output 1: {out[1].numpy()}")
        print(f"Outputs are different: {not torch.allclose(out[0], out[1], atol=1e-5)}")