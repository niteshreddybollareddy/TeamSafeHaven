#!/usr/bin/env python
"""Complete pipeline: preprocess -> train -> evaluate"""

import os
import sys
import argparse
from datetime import datetime
from preprocess import process_dataset
from train import train_model

def main():
    parser = argparse.ArgumentParser(description='Baby Cry Detection Pipeline')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocess dataset')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing step')
    args = parser.parse_args()
    
    print("="*60)
    print("Baby Cry Detection - Complete Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Preprocess
    if not args.skip_preprocess:
        print("\n[1/2] Preprocessing audio files...")
        if args.reprocess or not os.path.exists('processed_dataset'):
            process_dataset()
        else:
            print("Skipping preprocessing (use --reprocess to force)")
    else:
        print("\n[1/2] Skipping preprocessing...")
    
    # Step 2: Train
    print("\n[2/2] Training model...")
    train_model()
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("Model saved to: models/best_model.pth")
    print("="*60)

if __name__ == '__main__':
    main()
