import json
import numpy as np
from typing import List, Dict, Union
import pickle

class FeatureContract:
    """
    Defines the standard data format for landmarks across the pipeline.
    """
    
    @staticmethod
    def create_hand_data(hand_type: str, landmarks: List[Dict]) -> Dict:
        """
        Create standardized hand data structure.
        
        Args:
            hand_type: 'Left' or 'Right'
            landmarks: List of 21 landmark dicts with x, y, z, visibility
            
        Returns:
            Standardized hand data dict
        """
        assert hand_type in ['Left', 'Right'], "hand_type must be 'Left' or 'Right'"
        assert len(landmarks) == 21, "Hand must have exactly 21 landmarks"
        
        return {
            'type': hand_type,
            'landmarks': landmarks,
            'num_landmarks': len(landmarks)
        }
    
    @staticmethod
    def create_pose_data(upper_body: Dict, all_landmarks: List[Dict] = None) -> Dict:
        """
        Create standardized pose data structure.
        
        Args:
            upper_body: Dict of named upper body landmarks
            all_landmarks: Optional list of all 33 pose landmarks
            
        Returns:
            Standardized pose data dict
        """
        return {
            'upper_body': upper_body,
            'all_landmarks': all_landmarks or [],
            'num_landmarks': len(all_landmarks) if all_landmarks else len(upper_body)
        }
    
    @staticmethod
    def create_frame_data(frame_id: int, 
                         hands: List[Dict], 
                         pose: Dict = None,
                         timestamp: float = None) -> Dict:
        """
        Create standardized frame data structure.
        
        Args:
            frame_id: Frame number/ID
            hands: List of hand data dicts
            pose: Optional pose data dict
            timestamp: Optional timestamp in milliseconds
            
        Returns:
            Standardized frame data dict
        """
        return {
            'frame_id': frame_id,
            'timestamp': timestamp or frame_id * 33.33,  # Assume 30fps
            'hands': hands,
            'pose': pose,
            'num_hands': len(hands)
        }
    
    @staticmethod
    def validate_hand_data(hand_data: Dict) -> bool:
        """Validate hand data structure."""
        required_keys = ['type', 'landmarks']
        if not all(key in hand_data for key in required_keys):
            return False
        
        if hand_data['type'] not in ['Left', 'Right']:
            return False
        
        if len(hand_data['landmarks']) != 21:
            return False
        
        # Validate each landmark
        for lm in hand_data['landmarks']:
            if not all(key in lm for key in ['x', 'y', 'z']):
                return False
        
        return True
    
    @staticmethod
    def validate_frame_data(frame_data: Dict) -> bool:
        """Validate frame data structure."""
        required_keys = ['frame_id', 'hands']
        if not all(key in frame_data for key in required_keys):
            return False
        
        # Validate all hands
        for hand in frame_data['hands']:
            if not FeatureContract.validate_hand_data(hand):
                return False
        
        return True


class FeatureSerializer:
    """
    Handles serialization and deserialization of landmark data.
    """
    
    @staticmethod
    def to_json(data: Union[Dict, List], filepath: str) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def from_json(filepath: str) -> Union[Dict, List]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded data
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def to_numpy(landmarks: List[Dict]) -> np.ndarray:
        """
        Convert landmarks to numpy array.
        
        Args:
            landmarks: List of landmark dicts
            
        Returns:
            Numpy array of shape (num_landmarks, 3) for x, y, z
        """
        coords = []
        for lm in landmarks:
            coords.append([lm['x'], lm['y'], lm['z']])
        return np.array(coords)
    
    @staticmethod
    def from_numpy(array: np.ndarray) -> List[Dict]:
        """
        Convert numpy array back to landmark dicts.
        
        Args:
            array: Numpy array of shape (num_landmarks, 3)
            
        Returns:
            List of landmark dicts
        """
        landmarks = []
        for row in array:
            landmarks.append({
                'x': float(row[0]),
                'y': float(row[1]),
                'z': float(row[2]),
                'visibility': 1.0
            })
        return landmarks
    
    @staticmethod
    def sequence_to_numpy(sequence: List[List[Dict]]) -> np.ndarray:
        """
        Convert sequence of frames to numpy array.
        
        Args:
            sequence: List of frames, each containing list of hands
            
        Returns:
            Numpy array of shape (num_frames, num_hands, num_landmarks, 3)
        """
        frames = []
        for frame in sequence:
            hands = []
            for hand in frame:
                hand_array = FeatureSerializer.to_numpy(hand['landmarks'])
                hands.append(hand_array)
            
            # Pad to 2 hands if needed
            while len(hands) < 2:
                hands.append(np.zeros((21, 3)))
            
            frames.append(np.stack(hands[:2]))  # Take max 2 hands
        
        return np.array(frames)
    
    @staticmethod
    def save_pickle(data: any, filepath: str) -> None:
        """Save data using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(filepath: str) -> any:
        """Load data from pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class FeatureNormalizer:
    """
    Normalizes landmark features for model input.
    """
    
    @staticmethod
    def normalize_hand_to_wrist(hand_data: Dict) -> Dict:
        """
        Normalize hand landmarks relative to wrist (landmark 0).
        
        Args:
            hand_data: Hand data dict
            
        Returns:
            Normalized hand data
        """
        if len(hand_data['landmarks']) == 0:
            return hand_data
        
        wrist = hand_data['landmarks'][0]
        wrist_x, wrist_y, wrist_z = wrist['x'], wrist['y'], wrist['z']
        
        normalized_landmarks = []
        for lm in hand_data['landmarks']:
            normalized_landmarks.append({
                'x': lm['x'] - wrist_x,
                'y': lm['y'] - wrist_y,
                'z': lm['z'] - wrist_z,
                'visibility': lm.get('visibility', 1.0)
            })
        
        return {
            'type': hand_data['type'],
            'landmarks': normalized_landmarks
        }
    
    @staticmethod
    def scale_to_unit_box(landmarks: List[Dict]) -> List[Dict]:
        """
        Scale landmarks to fit in unit box [0, 1].
        
        Args:
            landmarks: List of landmarks
            
        Returns:
            Scaled landmarks
        """
        if len(landmarks) == 0:
            return landmarks
        
        # Find bounding box
        xs = [lm['x'] for lm in landmarks]
        ys = [lm['y'] for lm in landmarks]
        zs = [lm['z'] for lm in landmarks]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        
        range_x = max_x - min_x or 1.0
        range_y = max_y - min_y or 1.0
        range_z = max_z - min_z or 1.0
        
        scaled = []
        for lm in landmarks:
            scaled.append({
                'x': (lm['x'] - min_x) / range_x,
                'y': (lm['y'] - min_y) / range_y,
                'z': (lm['z'] - min_z) / range_z,
                'visibility': lm.get('visibility', 1.0)
            })
        
        return scaled


if __name__ == "__main__":
    print("Testing Feature Utilities...")
    
    # Test data contract
    mock_landmarks = [{'x': i * 0.1, 'y': i * 0.1, 'z': 0.0, 'visibility': 1.0} for i in range(21)]
    hand_data = FeatureContract.create_hand_data('Right', mock_landmarks)
    
    print(f"Created hand data: {FeatureContract.validate_hand_data(hand_data)}")
    
    # Test serialization
    print("\nTesting serialization...")
    array = FeatureSerializer.to_numpy(mock_landmarks)
    print(f"Numpy array shape: {array.shape}")
    
    back_to_dict = FeatureSerializer.from_numpy(array)
    print(f"Converted back: {len(back_to_dict)} landmarks")
    
    # Test normalization
    print("\nTesting normalization...")
    normalized = FeatureNormalizer.normalize_hand_to_wrist(hand_data)
    print(f"Normalized wrist position: x={normalized['landmarks'][0]['x']:.4f}")
    
    print("\nFeature utilities tests completed successfully!")
