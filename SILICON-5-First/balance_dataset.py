import os
import shutil
import random
from pathlib import Path

def balance_dataset(dataset_dir='Dataset', target_ratio=1.0):
    babycry_dir = os.path.join(dataset_dir, 'Baby_cry')
    dog_dir = os.path.join(dataset_dir, 'dog')
    excess_dir = os.path.join(dataset_dir, 'excess')
    
    # Count files (both .wav and .ogg)
    babycry_files = list(Path(babycry_dir).glob('*.wav')) + list(Path(babycry_dir).glob('*.ogg'))
    dog_files = list(Path(dog_dir).glob('*.wav')) + list(Path(dog_dir).glob('*.ogg'))
    
    print(f'Original distribution:')
    print(f'  Baby Cry: {len(babycry_files)}')
    print(f'  Dog: {len(dog_files)}')
    print(f'  Ratio: {len(babycry_files)/len(dog_files):.2f}:1')
    
    # Determine which class has more samples
    if len(babycry_files) > len(dog_files):
        excess_count = len(babycry_files) - len(dog_files)
        files_to_move = babycry_files
        source_dir = 'babycry'
    elif len(dog_files) > len(babycry_files):
        excess_count = len(dog_files) - len(babycry_files)
        files_to_move = dog_files
        source_dir = 'dog'
    else:
        print('\nDataset is already balanced.')
        return
    
    print(f'\nMoving {excess_count} files from {source_dir} to {excess_dir}')
    
    # Create excess directory
    os.makedirs(excess_dir, exist_ok=True)
    
    # Randomly select files to move
    random.shuffle(files_to_move)
    files_to_move = files_to_move[:excess_count]
    
    # Move files
    for file_path in files_to_move:
        dest = os.path.join(excess_dir, file_path.name)
        shutil.move(str(file_path), dest)
    
    # Verify
    remaining_babycry = len(list(Path(babycry_dir).glob('*.wav'))) + len(list(Path(babycry_dir).glob('*.ogg')))
    remaining_dog = len(list(Path(dog_dir).glob('*.wav'))) + len(list(Path(dog_dir).glob('*.ogg')))
    print(f'\nBalanced distribution:')
    print(f'  Baby Cry: {remaining_babycry}')
    print(f'  Dog: {remaining_dog}')
    print(f'  Ratio: {remaining_babycry/remaining_dog:.2f}:1')
    print(f'  Excess stored in: {excess_dir}')

if __name__ == '__main__':
    balance_dataset('Dataset', target_ratio=1.0)
