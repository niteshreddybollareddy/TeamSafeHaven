"""
Compare CNN and Transformer models, or ensemble their predictions
"""

import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path

# Add paths for both architectures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DualModelPredictor:
    def __init__(self, cnn_model_path, transformer_model_path):
        """Initialize both CNN and Transformer models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")
        
        # Load CNN model
        print("Loading CNN model...")
        cnn_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, cnn_base_path)
        from model import CryClassifier as CNNModel
        import config as cnn_config
        
        self.cnn_model = CNNModel().to(self.device)
        cnn_checkpoint = torch.load(cnn_model_path, map_location=self.device, weights_only=False)
        self.cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
        self.cnn_model.eval()
        self.cnn_stats = cnn_checkpoint['stats']
        print(f"✓ CNN loaded (13,074 params, 93.88% test acc)")
        
        # Load Transformer model
        print("\nLoading Transformer model...")
        transformer_base_path = os.path.dirname(os.path.dirname(transformer_model_path))
        import importlib.util
        spec = importlib.util.spec_from_file_location("transformer_model", os.path.join(transformer_base_path, "model.py"))
        transformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_module)
        TransformerClassifier = transformer_module.TransformerClassifier
        
        spec_config = importlib.util.spec_from_file_location("transformer_config", os.path.join(transformer_base_path, "config.py"))
        transformer_config = importlib.util.module_from_spec(spec_config)
        spec_config.loader.exec_module(transformer_config)
        
        self.transformer_model = TransformerClassifier(
            num_classes=transformer_config.NUM_CLASSES,
            d_model=transformer_config.D_MODEL,
            num_heads=transformer_config.NUM_HEADS,
            num_layers=transformer_config.NUM_LAYERS,
            dropout=transformer_config.DROPOUT
        ).to(self.device)
        transformer_checkpoint = torch.load(transformer_model_path, map_location=self.device, weights_only=False)
        self.transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
        self.transformer_model.eval()
        self.transformer_stats = transformer_checkpoint['stats']
        print(f"✓ Transformer loaded (15,066 params, 94.90% test acc)")
        
        self.cnn_config = cnn_config
        self.transformer_config = transformer_config
        
        print("\n✓ Both models loaded successfully!")
    
    def preprocess_for_cnn(self, audio_path):
        """Preprocess audio for CNN model"""
        y, sr = librosa.load(audio_path, sr=self.cnn_config.SAMPLE_RATE, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if np.abs(y_trimmed).max() > 0:
            y_trimmed = y_trimmed / np.abs(y_trimmed).max()
        
        mel_spec = librosa.feature.melspectrogram(
            y=y_trimmed, 
            sr=self.cnn_config.SAMPLE_RATE, 
            n_mels=self.cnn_config.N_MELS, 
            n_fft=self.cnn_config.N_FFT, 
            hop_length=self.cnn_config.HOP_LENGTH,
            fmin=50, fmax=8000, power=2.0
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Pad/truncate for CNN (fixed size)
        current_frames = mel_spec_db.shape[1]
        if current_frames < self.cnn_config.MAX_TIME_FRAMES:
            pad_width = self.cnn_config.MAX_TIME_FRAMES - current_frames
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
        else:
            start = (current_frames - self.cnn_config.MAX_TIME_FRAMES) // 2
            mel_spec_db = mel_spec_db[:, start:start + self.cnn_config.MAX_TIME_FRAMES]
        
        # Normalize
        mel_spec_norm = (mel_spec_db - self.cnn_stats['mean']) / (self.cnn_stats['std'] + 1e-8)
        return torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def preprocess_for_transformer(self, audio_path):
        """Preprocess audio for Transformer model"""
        y, sr = librosa.load(audio_path, sr=self.transformer_config.SAMPLE_RATE, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if np.abs(y_trimmed).max() > 0:
            y_trimmed = y_trimmed / np.abs(y_trimmed).max()
        
        mel_spec = librosa.feature.melspectrogram(
            y=y_trimmed, 
            sr=self.transformer_config.SAMPLE_RATE, 
            n_mels=self.transformer_config.N_MELS, 
            n_fft=self.transformer_config.N_FFT, 
            hop_length=self.transformer_config.HOP_LENGTH,
            fmin=50, fmax=8000, power=2.0
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Pad/truncate for Transformer
        if hasattr(self.transformer_config, 'MAX_TIME_FRAMES') and self.transformer_config.MAX_TIME_FRAMES is not None:
            current_frames = mel_spec_db.shape[1]
            if current_frames < self.transformer_config.MAX_TIME_FRAMES:
                pad_width = self.transformer_config.MAX_TIME_FRAMES - current_frames
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
            else:
                start = (current_frames - self.transformer_config.MAX_TIME_FRAMES) // 2
                mel_spec_db = mel_spec_db[:, start:start + self.transformer_config.MAX_TIME_FRAMES]
        
        # Normalize
        mel_spec_norm = (mel_spec_db - self.transformer_stats['mean']) / (self.transformer_stats['std'] + 1e-8)
        return torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def predict_single_model(self, model, mel_tensor):
        """Get prediction from a single model"""
        with torch.no_grad():
            output = model(mel_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
            cry_prob = probs[0][0].item()
            not_cry_prob = probs[0][1].item()
        
        return {
            'pred_class': pred_class,
            'confidence': confidence,
            'cry_prob': cry_prob,
            'not_cry_prob': not_cry_prob
        }
    
    def predict_comparison(self, audio_path):
        """Compare predictions from both models"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"\nProcessing: {audio_path}")
        
        # Preprocess for both models
        cnn_input = self.preprocess_for_cnn(audio_path)
        transformer_input = self.preprocess_for_transformer(audio_path)
        
        # Get predictions
        cnn_result = self.predict_single_model(self.cnn_model, cnn_input)
        transformer_result = self.predict_single_model(self.transformer_model, transformer_input)
        
        # Ensemble prediction (average probabilities)
        ensemble_cry_prob = (cnn_result['cry_prob'] + transformer_result['cry_prob']) / 2
        ensemble_not_cry_prob = (cnn_result['not_cry_prob'] + transformer_result['not_cry_prob']) / 2
        ensemble_pred_class = 0 if ensemble_cry_prob > ensemble_not_cry_prob else 1
        ensemble_confidence = max(ensemble_cry_prob, ensemble_not_cry_prob)
        
        return {
            'file': os.path.basename(audio_path),
            'cnn': {
                'prediction': self.cnn_config.CLASS_NAMES[cnn_result['pred_class']],
                'confidence': cnn_result['confidence'],
                'cry_prob': cnn_result['cry_prob'],
                'not_cry_prob': cnn_result['not_cry_prob']
            },
            'transformer': {
                'prediction': self.transformer_config.CLASS_NAMES[transformer_result['pred_class']],
                'confidence': transformer_result['confidence'],
                'cry_prob': transformer_result['cry_prob'],
                'not_cry_prob': transformer_result['not_cry_prob']
            },
            'ensemble': {
                'prediction': self.cnn_config.CLASS_NAMES[ensemble_pred_class],
                'confidence': ensemble_confidence,
                'cry_prob': ensemble_cry_prob,
                'not_cry_prob': ensemble_not_cry_prob
            }
        }
    
    def print_results(self, results):
        """Print comparison results in a nice format"""
        print("\n" + "="*70)
        print("PREDICTION COMPARISON")
        print("="*70)
        print(f"File: {results['file']}\n")
        
        print("CNN (93.88% accuracy):")
        print(f"  Prediction: {results['cnn']['prediction'].upper()}")
        print(f"  Confidence: {results['cnn']['confidence']:.1%}")
        print(f"  Probabilities: Cry={results['cnn']['cry_prob']:.1%}, Not Cry={results['cnn']['not_cry_prob']:.1%}")
        
        print("\nTransformer (94.90% accuracy) ⭐:")
        print(f"  Prediction: {results['transformer']['prediction'].upper()}")
        print(f"  Confidence: {results['transformer']['confidence']:.1%}")
        print(f"  Probabilities: Cry={results['transformer']['cry_prob']:.1%}, Not Cry={results['transformer']['not_cry_prob']:.1%}")
        
        print("\nEnsemble (Both models averaged):")
        print(f"  Prediction: {results['ensemble']['prediction'].upper()}")
        print(f"  Confidence: {results['ensemble']['confidence']:.1%}")
        print(f"  Probabilities: Cry={results['ensemble']['cry_prob']:.1%}, Not Cry={results['ensemble']['not_cry_prob']:.1%}")
        
        # Agreement indicator
        if results['cnn']['prediction'] == results['transformer']['prediction']:
            print("\n✓ Both models AGREE on the prediction")
        else:
            print("\n⚠️  Models DISAGREE - Ensemble used for final decision")
        
        print("="*70)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare CNN and Transformer predictions')
    parser.add_argument('audio_file', help='Audio file to process')
    parser.add_argument('--cnn-model', required=True, help='Path to CNN model')
    parser.add_argument('--transformer-model', required=True, help='Path to Transformer model')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = DualModelPredictor(args.cnn_model, args.transformer_model)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        sys.exit(1)
    
    # Get predictions
    try:
        results = predictor.predict_comparison(args.audio_file)
        predictor.print_results(results)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("="*70)
        print("Baby Cry Detector - Model Comparison Tool")
        print("="*70)
        print("\nThis script compares predictions from both CNN and Transformer models.")
        print("\nUsage:")
        print("  python compare_models.py <audio_file> \\")
        print("    --cnn-model path/to/cnn/best_model.pth \\")
        print("    --transformer-model path/to/transformer/best_model.pth")
        print("\nExample:")
        print("  python compare_models.py baby_cry.wav \\")
        print("    --cnn-model ../architecture1/models/best_model.pth \\")
        print("    --transformer-model ../architecture3/models/best_model.pth")
        sys.exit(0)
    else:
        main()


# python3 compare_model.py /Users/theepireddykarthikreddy/Desktop/SILICON/Bahadurpalle.wav --cnn-model /Users/theepireddykarthikreddy/Desktop/SILICON/models/best_model.pth --transformer-model /Users/theepireddykarthikreddy/Desktop/SILICON/other_architectures/architecture3/models/best_model.pth
