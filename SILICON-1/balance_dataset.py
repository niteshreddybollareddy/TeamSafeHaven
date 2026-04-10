import os
import shutil
import random
from pathlib import Path

def balance_dataset(dataset_dir='Dataset', target_ratio=1.0):
    cry_dir = os.path.join(dataset_dir, 'cry')
    not_cry_dir = os.path.join(dataset_dir, 'not_cry')
    excess_dir = os.path.join(dataset_dir, 'not_cry_excess')
    
    # Count files (both .wav and .ogg)
    cry_files = list(Path(cry_dir).glob('*.wav')) + list(Path(cry_dir).glob('*.ogg'))
    not_cry_files = list(Path(not_cry_dir).glob('*.wav')) + list(Path(not_cry_dir).glob('*.ogg'))
    
    print(f'Original distribution:')
    print(f'  Cry: {len(cry_files)}')
    print(f'  Not Cry: {len(not_cry_files)}')
    print(f'  Ratio: {len(cry_files)/len(not_cry_files):.2f}:1')
    
    # Calculate target number of not_cry files (match cry count)
    target_not_cry_count = len(cry_files)
    excess_count = len(not_cry_files) - target_not_cry_count
    
    if excess_count <= 0:
        print('\nDataset is already balanced or cry has more samples.')
        return
    
    print(f'\nTarget not_cry samples: {target_not_cry_count}')
    print(f'Moving {excess_count} files to {excess_dir}')
    
    # Create excess directory
    os.makedirs(excess_dir, exist_ok=True)
    
    # Randomly select files to move
    random.shuffle(not_cry_files)
    files_to_move = not_cry_files[:excess_count]
    
    # Move files
    for file_path in files_to_move:
        dest = os.path.join(excess_dir, file_path.name)
        shutil.move(str(file_path), dest)
    
    # Verify
    remaining_not_cry = len(list(Path(not_cry_dir).glob('*.wav'))) + len(list(Path(not_cry_dir).glob('*.ogg')))
    print(f'\nBalanced distribution:')
    print(f'  Cry: {len(cry_files)}')
    print(f'  Not Cry: {remaining_not_cry}')
    print(f'  Ratio: {len(cry_files)/remaining_not_cry:.2f}:1')
    print(f'  Excess stored in: {excess_dir}')

if __name__ == '__main__':
    balance_dataset('Dataset', target_ratio=1.0)
