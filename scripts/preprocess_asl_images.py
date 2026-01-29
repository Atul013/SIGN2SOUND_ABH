"""
Preprocess ASL Alphabet Image Dataset
Extracts hand landmarks from images and prepares data for training.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.hand_landmarks import HandLandmarks
from features.feature_utils import FeatureNormalizer, FeatureSerializer
from data.vocabulary import LETTER_TO_ID, NUM_CLASSES


def preprocess_asl_dataset(
    dataset_dir: str,
    output_dir: str,
    split: str = "train",
    max_samples_per_class: int = None
):
    """
    Preprocess ASL alphabet image dataset.
    
    Args:
        dataset_dir: Path to ASL dataset directory
        output_dir: Path to save processed data
        split: Dataset split ('train' or 'test')
        max_samples_per_class: Maximum samples per class (None = all)
    """
    
    print(f"\n{'='*70}")
    print(f"Preprocessing ASL Alphabet Dataset - {split.upper()} Split")
    print(f"{'='*70}\n")
    
    # Initialize hand landmark extractor
    print("Initializing MediaPipe Hand Landmarker...")
    extractor = HandLandmarks(
        model_path='models/hand_landmarker.task',
        num_hands=1,  # ASL alphabet uses one hand
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    normalizer = FeatureNormalizer
    serializer = FeatureSerializer()
    
    # Setup paths
    split_dir = os.path.join(dataset_dir, f"asl_alphabet_{split}")
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(split_dir) 
                        if os.path.isdir(os.path.join(split_dir, d))])
    
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}\n")
    
    # Statistics
    stats = {
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'per_class': {}
    }
    
    # Process each class
    for class_name in class_dirs:
        class_dir = os.path.join(split_dir, class_name)
        
        # Get class ID
        try:
            # Try uppercase first (for A-Z), then lowercase (for del, nothing, space)
            if class_name.upper() in LETTER_TO_ID:
                class_id = LETTER_TO_ID[class_name.upper()]
            elif class_name.lower() in LETTER_TO_ID:
                class_id = LETTER_TO_ID[class_name.lower()]
            else:
                print(f"⚠️  Skipping unknown class: {class_name}")
                continue
        except KeyError:
            print(f"⚠️  Skipping unknown class: {class_name}")
            continue
        
        # Get all images
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
        
        print(f"Processing class '{class_name}' (ID: {class_id}): {len(image_files)} images")
        
        class_stats = {
            'total': len(image_files),
            'successful': 0,
            'failed': 0
        }
        
        # Create output directory for this class
        class_output_dir = os.path.join(output_split_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"  {class_name}", leave=False):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    class_stats['failed'] += 1
                    continue
                
                # Extract landmarks
                detection_result, result = extractor.extract_landmarks(image)
                
                if not result or len(result) == 0:
                    class_stats['failed'] += 1
                    continue
                
                # Get first detected hand (assuming one hand per image)
                hand_data = result[0]
                
                # Normalize landmarks (wrist-relative)
                normalized_hand = normalizer.normalize_hand_to_wrist(hand_data)
                landmarks = normalized_hand['landmarks']
                
                # Convert to numpy array (21 landmarks * 3 coordinates = 63 features)
                features = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
                features = features.flatten()  # Shape: (63,)
                
                # Save as numpy file
                output_filename = os.path.splitext(img_file)[0] + '.npy'
                output_path = os.path.join(class_output_dir, output_filename)
                np.save(output_path, features)
                
                class_stats['successful'] += 1
                stats['successful'] += 1
                
            except Exception as e:
                print(f"    Error processing {img_file}: {str(e)}")
                class_stats['failed'] += 1
                stats['failed'] += 1
        
        stats['total_images'] += class_stats['total']
        stats['per_class'][class_name] = class_stats
        
        print(f"  ✓ {class_name}: {class_stats['successful']}/{class_stats['total']} successful\n")
    
    # Save statistics
    stats_file = os.path.join(output_split_dir, 'preprocessing_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Preprocessing Complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful: {stats['successful']} ({stats['successful']/stats['total_images']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total_images']*100:.1f}%)")
    print(f"\nProcessed data saved to: {output_split_dir}")
    print(f"Statistics saved to: {stats_file}")
    print(f"{'='*70}\n")
    
    return stats


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
              if os.path.isdir(os.path.join(train_dir, d)) and d != 'val']
    
    for class_name in classes:
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)
        
        # Get all processed files
        files = [f for f in os.listdir(class_train_dir) if f.endswith('.npy')]
        
        # Shuffle and split
        np.random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        
        val_files = files[split_idx:]
        
        # Move validation files
        for f in val_files:
            src = os.path.join(class_train_dir, f)
            dst = os.path.join(class_val_dir, f)
            os.rename(src, dst)
        
        print(f"{class_name}: {split_idx} train, {len(val_files)} val")
    
    print(f"\n✓ Splits created successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess ASL Alphabet Dataset")
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=r'E:\Projects\S2S\ASL\ASL_Alphabet_Dataset',
        help='Path to ASL dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per class (for testing)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Process only training set'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create train/val splits from training data'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Process training set
    print("Processing training set...")
    train_stats = preprocess_asl_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        split='train',
        max_samples_per_class=args.max_samples
    )
    
    # Process test set
    if not args.train_only:
        print("\nProcessing test set...")
        test_stats = preprocess_asl_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split='test',
            max_samples_per_class=args.max_samples
        )
    
    # Create train/val splits
    if args.create_splits:
        create_dataset_splits(args.output_dir, train_ratio=0.85)
    
    print("\n✅ All preprocessing complete!")
