"""
Data augmentation functions for landmark sequences.
Applies spatial and temporal transformations to improve model generalization.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import random


class LandmarkAugmenter:
    """
    Applies data augmentation to landmark sequences for training robustness.
    """
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 translation_range: float = 0.05,
                 temporal_jitter_range: int = 2,
                 noise_std: float = 0.01,
                 augmentation_probability: float = 0.5):
        """
        Initialize the augmenter with transformation parameters.
        
        Args:
            rotation_range: Maximum rotation in degrees (±)
            scale_range: Min and max scale factors
            translation_range: Maximum translation as fraction of frame size
            temporal_jitter_range: Maximum frames to shift temporally
            noise_std: Standard deviation of Gaussian noise
            augmentation_probability: Probability of applying each augmentation
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.temporal_jitter_range = temporal_jitter_range
        self.noise_std = noise_std
        self.augmentation_probability = augmentation_probability
    
    def apply_rotation(self, landmarks: List[Dict], angle: Optional[float] = None) -> List[Dict]:
        """
        Apply 2D rotation to landmarks around the wrist center.
        
        Args:
            landmarks: List of landmark dicts with x, y, z coordinates
            angle: Rotation angle in degrees (random if None)
            
        Returns:
            Rotated landmarks
        """
        if angle is None:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # Convert to radians
        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Rotation matrix
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        rotated_landmarks = []
        for lm in landmarks:
            # Get center (wrist is typically landmark 0)
            center_x = landmarks[0]['x']
            center_y = landmarks[0]['y']
            
            # Translate to origin
            x_centered = lm['x'] - center_x
            y_centered = lm['y'] - center_y
            
            # Apply rotation
            rotated = rotation_matrix @ np.array([x_centered, y_centered])
            
            # Translate back
            rotated_landmarks.append({
                'x': rotated[0] + center_x,
                'y': rotated[1] + center_y,
                'z': lm['z']  # Keep z unchanged for 2D rotation
            })
        
        return rotated_landmarks
    
    def apply_scale(self, landmarks: List[Dict], scale: Optional[float] = None) -> List[Dict]:
        """
        Apply scaling to landmarks relative to wrist center.
        
        Args:
            landmarks: List of landmark dicts
            scale: Scale factor (random if None)
            
        Returns:
            Scaled landmarks
        """
        if scale is None:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        scaled_landmarks = []
        center_x = landmarks[0]['x']
        center_y = landmarks[0]['y']
        center_z = landmarks[0]['z']
        
        for lm in landmarks:
            scaled_landmarks.append({
                'x': center_x + (lm['x'] - center_x) * scale,
                'y': center_y + (lm['y'] - center_y) * scale,
                'z': center_z + (lm['z'] - center_z) * scale
            })
        
        return scaled_landmarks
    
    def apply_translation(self, landmarks: List[Dict], 
                         dx: Optional[float] = None, 
                         dy: Optional[float] = None) -> List[Dict]:
        """
        Apply translation to all landmarks.
        
        Args:
            landmarks: List of landmark dicts
            dx: X translation (random if None)
            dy: Y translation (random if None)
            
        Returns:
            Translated landmarks
        """
        if dx is None:
            dx = random.uniform(-self.translation_range, self.translation_range)
        if dy is None:
            dy = random.uniform(-self.translation_range, self.translation_range)
        
        translated_landmarks = []
        for lm in landmarks:
            translated_landmarks.append({
                'x': lm['x'] + dx,
                'y': lm['y'] + dy,
                'z': lm['z']
            })
        
        return translated_landmarks
    
    def apply_noise(self, landmarks: List[Dict], std: Optional[float] = None) -> List[Dict]:
        """
        Add Gaussian noise to landmark coordinates.
        
        Args:
            landmarks: List of landmark dicts
            std: Standard deviation of noise (uses default if None)
            
        Returns:
            Noisy landmarks
        """
        if std is None:
            std = self.noise_std
        
        noisy_landmarks = []
        for lm in landmarks:
            noisy_landmarks.append({
                'x': lm['x'] + np.random.normal(0, std),
                'y': lm['y'] + np.random.normal(0, std),
                'z': lm['z'] + np.random.normal(0, std)
            })
        
        return noisy_landmarks
    
    def apply_temporal_jitter(self, sequence: List[List[Dict]]) -> List[List[Dict]]:
        """
        Apply temporal jittering by randomly dropping or duplicating frames.
        
        Args:
            sequence: List of frames, each containing landmark dicts
            
        Returns:
            Temporally jittered sequence
        """
        if len(sequence) < 3:
            return sequence
        
        jittered = []
        for i, frame in enumerate(sequence):
            # Randomly skip frames (except first and last)
            if i > 0 and i < len(sequence) - 1:
                if random.random() < 0.1:  # 10% chance to skip
                    continue
            
            jittered.append(frame)
            
            # Randomly duplicate frames
            if random.random() < 0.05:  # 5% chance to duplicate
                jittered.append(frame)
        
        return jittered
    
    def augment_frame(self, frame_hands: List[Dict]) -> List[Dict]:
        """
        Apply spatial augmentations to a single frame (all hands).
        
        Args:
            frame_hands: List of hand dicts, each with 'type' and 'landmarks'
            
        Returns:
            Augmented frame hands
        """
        augmented_hands = []
        
        # Apply same transformations to all hands in frame
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        dx = random.uniform(-self.translation_range, self.translation_range)
        dy = random.uniform(-self.translation_range, self.translation_range)
        
        for hand in frame_hands:
            landmarks = hand['landmarks']
            
            # Apply augmentations with probability
            if random.random() < self.augmentation_probability:
                landmarks = self.apply_rotation(landmarks, angle)
            
            if random.random() < self.augmentation_probability:
                landmarks = self.apply_scale(landmarks, scale)
            
            if random.random() < self.augmentation_probability:
                landmarks = self.apply_translation(landmarks, dx, dy)
            
            if random.random() < self.augmentation_probability:
                landmarks = self.apply_noise(landmarks)
            
            augmented_hands.append({
                'type': hand['type'],
                'landmarks': landmarks
            })
        
        return augmented_hands
    
    def augment_sequence(self, sequence: List[List[Dict]], 
                        apply_temporal: bool = True) -> List[List[Dict]]:
        """
        Apply augmentations to an entire sequence.
        
        Args:
            sequence: List of frames, each containing hand landmark dicts
            apply_temporal: Whether to apply temporal jittering
            
        Returns:
            Augmented sequence
        """
        # Apply spatial augmentations to each frame
        augmented = [self.augment_frame(frame) for frame in sequence]
        
        # Apply temporal augmentation
        if apply_temporal and random.random() < self.augmentation_probability:
            augmented = self.apply_temporal_jitter(augmented)
        
        return augmented
    
    def augment_batch(self, sequences: List[List[List[Dict]]]) -> List[List[List[Dict]]]:
        """
        Apply augmentations to a batch of sequences.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Augmented sequences
        """
        return [self.augment_sequence(seq) for seq in sequences]


class MirrorAugmenter:
    """
    Applies horizontal mirroring to create left-right symmetric augmentations.
    Useful for sign language where many signs have symmetric variants.
    """
    
    @staticmethod
    def mirror_landmarks(landmarks: List[Dict]) -> List[Dict]:
        """
        Mirror landmarks horizontally (flip x-coordinates).
        
        Args:
            landmarks: List of landmark dicts
            
        Returns:
            Mirrored landmarks
        """
        mirrored = []
        for lm in landmarks:
            mirrored.append({
                'x': 1.0 - lm['x'],  # Flip x (assuming normalized 0-1)
                'y': lm['y'],
                'z': lm['z']
            })
        return mirrored
    
    @staticmethod
    def mirror_frame(frame_hands: List[Dict]) -> List[Dict]:
        """
        Mirror a frame and swap left/right hand labels.
        
        Args:
            frame_hands: List of hand dicts
            
        Returns:
            Mirrored frame with swapped hand types
        """
        mirrored_hands = []
        for hand in frame_hands:
            mirrored_landmarks = MirrorAugmenter.mirror_landmarks(hand['landmarks'])
            
            # Swap hand type
            hand_type = hand['type']
            if hand_type == 'Left':
                hand_type = 'Right'
            elif hand_type == 'Right':
                hand_type = 'Left'
            
            mirrored_hands.append({
                'type': hand_type,
                'landmarks': mirrored_landmarks
            })
        
        return mirrored_hands
    
    @staticmethod
    def mirror_sequence(sequence: List[List[Dict]]) -> List[List[Dict]]:
        """
        Mirror an entire sequence.
        
        Args:
            sequence: List of frames
            
        Returns:
            Mirrored sequence
        """
        return [MirrorAugmenter.mirror_frame(frame) for frame in sequence]


if __name__ == "__main__":
    # Test augmentation
    print("Testing Landmark Augmentation...")
    
    # Create mock hand landmarks
    mock_hand = {
        'type': 'Right',
        'landmarks': [{'x': 0.5 + i*0.01, 'y': 0.5 + i*0.01, 'z': 0.0} for i in range(21)]
    }
    
    mock_sequence = [[mock_hand] for _ in range(20)]
    
    # Test augmenter
    augmenter = LandmarkAugmenter()
    augmented_seq = augmenter.augment_sequence(mock_sequence)
    
    print(f"Original sequence length: {len(mock_sequence)}")
    print(f"Augmented sequence length: {len(augmented_seq)}")
    print(f"Original first landmark: x={mock_sequence[0][0]['landmarks'][0]['x']:.4f}")
    print(f"Augmented first landmark: x={augmented_seq[0][0]['landmarks'][0]['x']:.4f}")
    
    # Test mirroring
    print("\nTesting Mirror Augmentation...")
    mirrored_seq = MirrorAugmenter.mirror_sequence(mock_sequence)
    print(f"Original hand type: {mock_sequence[0][0]['type']}")
    print(f"Mirrored hand type: {mirrored_seq[0][0]['type']}")
    print(f"Original x: {mock_sequence[0][0]['landmarks'][0]['x']:.4f}")
    print(f"Mirrored x: {mirrored_seq[0][0]['landmarks'][0]['x']:.4f}")
    
    print("\n✅ Augmentation tests complete!")
