"""
Copy representative ASL alphabet images from dataset to UI static folder
"""
import os
import shutil
from pathlib import Path

# Source dataset directory
dataset_dir = Path('../ASL_Alphabet_Dataset/asl_alphabet_train')

# Destination directory
dest_dir = Path('../ui/static/images/asl_alphabet')
dest_dir.mkdir(parents=True, exist_ok=True)

# Letters to copy
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
special = ['del', 'nothing', 'space']

print("Copying ASL alphabet images...")
print("=" * 60)

copied = 0
for letter in letters:
    source_folder = dataset_dir / letter
    if source_folder.exists():
        # Get first image from folder (they're all similar)
        images = list(source_folder.glob('*.jpg'))
        if images:
            # Copy first image
            source_image = images[0]
            dest_image = dest_dir / f'{letter}.jpg'
            shutil.copy2(source_image, dest_image)
            print(f"[OK] Copied {letter}.jpg")
            copied += 1
    else:
        print(f"[SKIP] Folder not found: {letter}")

# Copy special gestures
for gesture in special:
    source_folder = dataset_dir / gesture
    if source_folder.exists():
        images = list(source_folder.glob('*.jpg'))
        if images:
            source_image = images[0]
            dest_image = dest_dir / f'{gesture}.jpg'
            shutil.copy2(source_image, dest_image)
            print(f"[OK] Copied {gesture}.jpg")
            copied += 1
    else:
        print(f"[SKIP] Folder not found: {gesture}")

print("=" * 60)
print(f"Total images copied: {copied}")
print(f"Destination: {dest_dir.absolute()}")
