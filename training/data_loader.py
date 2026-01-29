"""
Data Loader for ASL Alphabet Recognition

This module provides data loading utilities for training the ASL alphabet recognition model.
It loads preprocessed landmark sequences from Developer A's pipeline and prepares them for training.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
import random
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocabulary import LETTER_TO_ID, ID_TO_LETTER, NUM_CLASSES


class ASLDataset:
    """
    Dataset class for ASL alphabet recognition.
    Loads preprocessed landmark sequences and provides batching functionality.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        augment: bool = True,
        max_sequence_length: int = 30,
        feature_dim: int = 63  # 21 hand landmarks * 3 (x, y, z)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing processed data
            split: One of 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            max_sequence_length: Maximum sequence length (pad/truncate)
            feature_dim: Dimension of feature vector per frame
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == 'train')
        self.max_sequence_length = max_sequence_length
        self.feature_dim = feature_dim
        
        # Load data
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _load_samples(self) -> List[Dict]:
        """Load all samples for this split."""
        samples = []
        split_file = self.data_dir / 'splits' / f'{self.split}.txt'
        
        if split_file.exists():
            # Load from split file
            with open(split_file, 'r') as f:
                file_paths = [line.strip() for line in f if line.strip()]
            
            for file_path in file_paths:
                full_path = self.data_dir / 'processed' / self.split / file_path
                if full_path.exists():
                    samples.append(self._load_sample(full_path))
        else:
            # Load all files from processed directory
            processed_dir = self.data_dir / 'processed' / self.split
            if processed_dir.exists():
                for letter_dir in sorted(processed_dir.iterdir()):
                    if letter_dir.is_dir():
                        letter = letter_dir.name
                        if letter in LETTER_TO_ID:
                            for json_file in letter_dir.glob('*.json'):
                                samples.append(self._load_sample(json_file))
        
        return samples
    
    def _load_sample(self, file_path: Path) -> Dict:
        """Load a single sample from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract letter label from file path or data
        if 'label' in data:
            letter = data['label']
        else:
            # Infer from directory name
            letter = file_path.parent.name
        
        return {
            'file_path': str(file_path),
            'letter': letter,
            'label': LETTER_TO_ID[letter],
            'landmarks': data.get('landmarks', data.get('hand_landmarks', [])),
            'metadata': data.get('metadata', {})
        }
    
    def _extract_features(self, landmarks: List) -> np.ndarray:
        """
        Extract feature vector from landmarks.
        
        Args:
            landmarks: List of landmark frames
            
        Returns:
            Feature array of shape (T, feature_dim)
        """
        features = []
        
        for frame in landmarks:
            if isinstance(frame, dict):
                # Extract hand landmarks
                hand_landmarks = frame.get('hand_landmarks', frame.get('landmarks', {}))
                
                # Flatten landmarks to feature vector
                frame_features = []
                if isinstance(hand_landmarks, dict):
                    # Assuming format: {0: {'x': ..., 'y': ..., 'z': ...}, ...}
                    for i in range(21):  # 21 hand landmarks
                        if i in hand_landmarks or str(i) in hand_landmarks:
                            lm = hand_landmarks.get(i, hand_landmarks.get(str(i), {}))
                            frame_features.extend([
                                lm.get('x', 0.0),
                                lm.get('y', 0.0),
                                lm.get('z', 0.0)
                            ])
                        else:
                            frame_features.extend([0.0, 0.0, 0.0])
                elif isinstance(hand_landmarks, list):
                    # Assuming format: [[x, y, z], ...]
                    for lm in hand_landmarks[:21]:
                        if isinstance(lm, (list, tuple)) and len(lm) >= 3:
                            frame_features.extend(lm[:3])
                        elif isinstance(lm, dict):
                            frame_features.extend([
                                lm.get('x', 0.0),
                                lm.get('y', 0.0),
                                lm.get('z', 0.0)
                            ])
                
                if len(frame_features) == self.feature_dim:
                    features.append(frame_features)
            elif isinstance(frame, (list, np.ndarray)):
                # Already flattened features
                if len(frame) == self.feature_dim:
                    features.append(frame)
        
        if not features:
            # Return zero features if no valid landmarks
            features = [[0.0] * self.feature_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """
        Pad or truncate sequence to max_sequence_length.
        
        Args:
            features: Feature array of shape (T, feature_dim)
            
        Returns:
            Padded/truncated array of shape (max_sequence_length, feature_dim)
        """
        seq_len = len(features)
        
        if seq_len >= self.max_sequence_length:
            # Truncate
            return features[:self.max_sequence_length]
        else:
            # Pad with zeros
            padding = np.zeros((self.max_sequence_length - seq_len, self.feature_dim), dtype=np.float32)
            return np.vstack([features, padding])
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to features.
        
        Args:
            features: Feature array of shape (T, feature_dim)
            
        Returns:
            Augmented feature array
        """
        if not self.augment:
            return features
        
        # Random scaling (simulate different hand sizes)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            features = features * scale
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features = features + noise
        
        # Random temporal shift (for sequences)
        if random.random() < 0.3 and len(features) > 5:
            shift = random.randint(-2, 2)
            if shift > 0:
                features = np.vstack([features[shift:], np.zeros((shift, self.feature_dim))])
            elif shift < 0:
                features = np.vstack([np.zeros((-shift, self.feature_dim)), features[:shift]])
        
        return features
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label, sequence_length)
        """
        sample = self.samples[idx]
        
        # Extract features
        features = self._extract_features(sample['landmarks'])
        
        # Store original sequence length
        seq_len = min(len(features), self.max_sequence_length)
        
        # Apply augmentation
        features = self._augment_features(features)
        
        # Pad or truncate
        features = self._pad_or_truncate(features)
        
        return features, sample['label'], seq_len
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a batch of samples.
        
        Args:
            batch_size: Number of samples in batch
            shuffle: Whether to shuffle samples
            
        Returns:
            Tuple of (batch_features, batch_labels, batch_seq_lengths)
        """
        if shuffle:
            indices = random.sample(range(len(self)), min(batch_size, len(self)))
        else:
            indices = list(range(min(batch_size, len(self))))
        
        batch_features = []
        batch_labels = []
        batch_seq_lengths = []
        
        for idx in indices:
            features, label, seq_len = self[idx]
            batch_features.append(features)
            batch_labels.append(label)
            batch_seq_lengths.append(seq_len)
        
        return (
            np.array(batch_features, dtype=np.float32),
            np.array(batch_labels, dtype=np.int64),
            np.array(batch_seq_lengths, dtype=np.int32)
        )


class DataLoader:
    """
    Data loader with batching and shuffling support.
    """
    
    def __init__(self, dataset: ASLDataset, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize data loader.
        
        Args:
            dataset: ASLDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = (len(dataset) + batch_size - 1) // batch_size
        
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches
    
    def __iter__(self):
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            batch_features = []
            batch_labels = []
            batch_seq_lengths = []
            
            for idx in batch_indices:
                features, label, seq_len = self.dataset[idx]
                batch_features.append(features)
                batch_labels.append(label)
                batch_seq_lengths.append(seq_len)
            
            yield (
                np.array(batch_features, dtype=np.float32),
                np.array(batch_labels, dtype=np.int64),
                np.array(batch_seq_lengths, dtype=np.int32)
            )


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    max_sequence_length: int = 30
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory containing processed data
        batch_size: Batch size
        max_sequence_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = ASLDataset(
        data_dir,
        split='train',
        augment=True,
        max_sequence_length=max_sequence_length
    )
    
    val_dataset = ASLDataset(
        data_dir,
        split='val',
        augment=False,
        max_sequence_length=max_sequence_length
    )
    
    test_dataset = ASLDataset(
        data_dir,
        split='test',
        augment=False,
        max_sequence_length=max_sequence_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Demo usage
    print("ASL Alphabet Data Loader Demo")
    print("=" * 60)
    
    # Example: Create data loaders
    data_dir = "data"
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir,
            batch_size=16,
            max_sequence_length=30
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Get a sample batch
        if len(train_loader) > 0:
            for features, labels, seq_lengths in train_loader:
                print(f"\nSample batch:")
                print(f"  Features shape: {features.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Sequence lengths shape: {seq_lengths.shape}")
                print(f"  Sample labels: {[ID_TO_LETTER[l] for l in labels[:5]]}")
                break
    except Exception as e:
        print(f"\nNote: {e}")
        print("This is expected if data hasn't been prepared yet.")
        print("\nTo use this data loader:")
        print("1. Download ASL alphabet dataset")
        print("2. Preprocess with Developer A's pipeline")
        print("3. Organize in data/processed/train, val, test directories")
