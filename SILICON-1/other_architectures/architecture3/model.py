import torch
import torch.nn as nn
import torch.nn.init as init
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding - increased max_len to handle longer sequences
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=2, d_model=28, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection: (B, 1, 64, T) -> (B, T, 64) -> (B, T, d_model)
        self.input_proj = nn.Linear(64, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding - increased max_len to 512 for longer sequences
        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        
        # Transformer encoder - optimized configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,  # 2x keeps us under budget
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly for transformers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for transformers
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 1, 64, T) where T can be variable
        Returns:
            Output tensor of shape (B, num_classes)
        """
        B = x.size(0)
        
        # Reshape: (B, 1, 64, T) -> (B, T, 64)
        x = x.squeeze(1).permute(0, 2, 1)
        
        # Project to d_model: (B, T, 64) -> (B, T, d_model)
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Add positional encoding (handles variable length)
        x = self.pos_encoding(x)
        
        # Transformer encoding: (B, T, d_model) -> (B, T, d_model)
        x = self.transformer(x)
        
        # Global mean pooling over time: (B, T, d_model) -> (B, d_model)
        x = x.mean(dim=1)
        
        # Classifier: (B, d_model) -> (B, num_classes)
        x = self.classifier(x)
        
        return x

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_initialization():
    """Test that model produces different outputs for different inputs"""
    model = TransformerClassifier()
    model.eval()
    
    # Create two different random inputs
    x1 = torch.randn(1, 1, 64, 128)
    x2 = torch.randn(1, 1, 64, 128)
    
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
    # Optimized configuration - uses ~88% of parameter budget
    TEST_D_MODEL = 28  # Increased from 20
    TEST_NUM_HEADS = 4  # 28/4 = 7 dims per head
    TEST_NUM_LAYERS = 2
    TEST_DROPOUT = 0.2
    TEST_NUM_CLASSES = 2
    TEST_MAX_PARAMS = 17000
    
    model = TransformerClassifier(
        num_classes=TEST_NUM_CLASSES,
        d_model=TEST_D_MODEL,
        num_heads=TEST_NUM_HEADS,
        num_layers=TEST_NUM_LAYERS,
        dropout=TEST_DROPOUT
    )
    
    params = count_parameters(model)
    print(f'Total parameters: {params:,}')
    print(f'Parameter budget: {TEST_MAX_PARAMS:,}')
    
    if params <= TEST_MAX_PARAMS:
        print(f'✓ Within budget! ({TEST_MAX_PARAMS - params:,} parameters remaining)')
    else:
        print(f'❌ Exceeds budget by {params - TEST_MAX_PARAMS:,} parameters')
    
    # Test forward pass with fixed length
    print('\n--- Testing Fixed Length Input ---')
    x_fixed = torch.randn(2, 1, 64, 128)
    out = model(x_fixed)
    print(f'Input shape: {x_fixed.shape}')
    print(f'Output shape: {out.shape}')
    
    # Test forward pass with variable length
    print('\n--- Testing Variable Length Input ---')
    x_var = torch.randn(2, 1, 64, 200)
    out_var = model(x_var)
    print(f'Input shape: {x_var.shape}')
    print(f'Output shape: {out_var.shape}')
    
    # Test initialization
    print('\n--- Testing Model Initialization ---')
    test_model_initialization()
    
    # Check that different batches produce different outputs
    print("\n--- Testing Batch Outputs ---")
    model.eval()
    with torch.no_grad():
        out = model(x_fixed)
        print(f"Batch output 0: {out[0].numpy()}")
        print(f"Batch output 1: {out[1].numpy()}")
        print(f"Outputs are different: {not torch.allclose(out[0], out[1], atol=1e-5)}")