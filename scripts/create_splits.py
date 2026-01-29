"""
Create train/validation splits from already processed data
"""
import os
import numpy as np
import shutil

def create_dataset_splits(processed_dir: str, train_ratio: float = 0.85):
    """
    Create train/val splits from processed training data.
    
    Args:
        processed_dir: Directory with processed data
        train_ratio: Ratio of training data (rest goes to validation)
    """
    print(f"\n{'='*70}")
    print("Creating Train/Validation Splits")
    print(f"{'='*70}\n")
    
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all classes
    classes = [d for d in os.listdir(train_dir) 
              if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"Found {len(classes)} classes\n")
    
    for class_name in classes:
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)
        
        # Get all processed files
        files = [f for f in os.listdir(class_train_dir) if f.endswith('.npy')]
        
        if len(files) == 0:
            print(f"{class_name}: No files found, skipping")
            continue
        
        # Shuffle and split
        np.random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        
        val_files = files[split_idx:]
        
        # Move validation files
        moved = 0
        for f in val_files:
            src = os.path.join(class_train_dir, f)
            dst = os.path.join(class_val_dir, f)
            
            # Remove destination if it exists
            if os.path.exists(dst):
                os.remove(dst)
            
            # Move file
            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1
        
        print(f"{class_name}: {split_idx} train, {moved} val")
    
    print(f"\n✓ Splits created successfully!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create splits
    create_dataset_splits('data/processed', train_ratio=0.85)
    
    print("✅ Split creation complete!")
