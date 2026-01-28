import numpy as np
from typing import List, Dict, Optional
from collections import deque

class LandmarkFilter:
    """
    Filters and smooths landmark data to reduce noise and jitter.
    """
    
    def __init__(self, window_size: int = 5, alpha: float = 0.3):
        """
        Initialize the landmark filter.
        
        Args:
            window_size: Size of moving average window
            alpha: Smoothing factor for exponential moving average (0-1)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.ema_state = None
    
    def moving_average(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Apply moving average filter to landmarks.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            Smoothed landmarks
        """
        self.history.append(landmarks)
        
        if len(self.history) < 2:
            return landmarks
        
        # Average across all frames in history
        smoothed = []
        for hand_idx in range(len(landmarks)):
            hand_type = landmarks[hand_idx]['type']
            smoothed_landmarks = []
            
            for lm_idx in range(len(landmarks[hand_idx]['landmarks'])):
                avg_x, avg_y, avg_z = 0.0, 0.0, 0.0
                count = 0
                
                for hist_frame in self.history:
                    if hand_idx < len(hist_frame):
                        if hist_frame[hand_idx]['type'] == hand_type:
                            lm = hist_frame[hand_idx]['landmarks'][lm_idx]
                            avg_x += lm['x']
                            avg_y += lm['y']
                            avg_z += lm['z']
                            count += 1
                
                if count > 0:
                    smoothed_landmarks.append({
                        'x': avg_x / count,
                        'y': avg_y / count,
                        'z': avg_z / count,
                        'visibility': landmarks[hand_idx]['landmarks'][lm_idx].get('visibility', 1.0)
                    })
                else:
                    smoothed_landmarks.append(landmarks[hand_idx]['landmarks'][lm_idx])
            
            smoothed.append({
                'type': hand_type,
                'landmarks': smoothed_landmarks
            })
        
        return smoothed
    
    def exponential_moving_average(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Apply exponential moving average for smoother transitions.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            Smoothed landmarks
        """
        if self.ema_state is None:
            self.ema_state = landmarks
            return landmarks
        
        smoothed = []
        for hand_idx in range(len(landmarks)):
            if hand_idx >= len(self.ema_state):
                smoothed.append(landmarks[hand_idx])
                continue
            
            hand_type = landmarks[hand_idx]['type']
            smoothed_landmarks = []
            
            for lm_idx in range(len(landmarks[hand_idx]['landmarks'])):
                curr_lm = landmarks[hand_idx]['landmarks'][lm_idx]
                prev_lm = self.ema_state[hand_idx]['landmarks'][lm_idx]
                
                smoothed_landmarks.append({
                    'x': self.alpha * curr_lm['x'] + (1 - self.alpha) * prev_lm['x'],
                    'y': self.alpha * curr_lm['y'] + (1 - self.alpha) * prev_lm['y'],
                    'z': self.alpha * curr_lm['z'] + (1 - self.alpha) * prev_lm['z'],
                    'visibility': curr_lm.get('visibility', 1.0)
                })
            
            smoothed.append({
                'type': hand_type,
                'landmarks': smoothed_landmarks
            })
        
        self.ema_state = smoothed
        return smoothed
    
    def reset(self):
        """Reset filter state."""
        self.history.clear()
        self.ema_state = None


class OcclusionHandler:
    """
    Handles missing landmarks and occlusions through interpolation.
    """
    
    def __init__(self, 
                 min_confidence: float = 0.5,
                 max_interpolation_gap: int = 5):
        """
        Initialize occlusion handler.
        
        Args:
            min_confidence: Minimum confidence to consider landmark valid
            max_interpolation_gap: Maximum frames to interpolate over
        """
        self.min_confidence = min_confidence
        self.max_interpolation_gap = max_interpolation_gap
        self.last_valid_landmarks = {}
        self.missing_count = {}
    
    def is_valid_landmark(self, landmark: Dict) -> bool:
        """Check if landmark has sufficient confidence."""
        visibility = landmark.get('visibility', 1.0)
        return visibility >= self.min_confidence
    
    def interpolate_missing(self, 
                           current_landmarks: Optional[List[Dict]], 
                           hand_type: str) -> Optional[List[Dict]]:
        """
        Interpolate missing landmarks based on last valid data.
        
        Args:
            current_landmarks: Current frame landmarks (may be None/incomplete)
            hand_type: 'Left' or 'Right'
            
        Returns:
            Interpolated landmarks or None if gap too large
        """
        key = hand_type
        
        # If we have valid current landmarks, update cache
        if current_landmarks is not None:
            valid_count = sum(1 for lm in current_landmarks if self.is_valid_landmark(lm))
            
            if valid_count > len(current_landmarks) * 0.7:  # At least 70% valid
                self.last_valid_landmarks[key] = current_landmarks
                self.missing_count[key] = 0
                return current_landmarks
        
        # Track missing frames
        self.missing_count[key] = self.missing_count.get(key, 0) + 1
        
        # If gap is too large, give up
        if self.missing_count[key] > self.max_interpolation_gap:
            return None
        
        # Return last valid landmarks if available
        return self.last_valid_landmarks.get(key)
    
    def process_frame(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Process a frame, handling occlusions.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            Processed landmarks with interpolation applied
        """
        processed = []
        
        # Track which hand types we've seen
        seen_types = {hand['type'] for hand in landmarks}
        
        # Process existing hands
        for hand in landmarks:
            hand_landmarks = hand['landmarks']
            interpolated = self.interpolate_missing(hand_landmarks, hand['type'])
            
            if interpolated is not None:
                processed.append({
                    'type': hand['type'],
                    'landmarks': interpolated
                })
        
        # Check for completely missing hands
        for hand_type in ['Left', 'Right']:
            if hand_type not in seen_types:
                interpolated = self.interpolate_missing(None, hand_type)
                if interpolated is not None:
                    processed.append({
                        'type': hand_type,
                        'landmarks': interpolated
                    })
        
        return processed
    
    def reset(self):
        """Reset handler state."""
        self.last_valid_landmarks.clear()
        self.missing_count.clear()


class CameraMotionCompensator:
    """
    Compensates for camera movement by normalizing landmarks.
    """
    
    def __init__(self, reference_point_idx: int = 0):
        """
        Initialize motion compensator.
        
        Args:
            reference_point_idx: Index of landmark to use as reference (default: wrist)
        """
        self.reference_point_idx = reference_point_idx
        self.baseline_position = None
    
    def normalize_to_reference(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Normalize landmarks relative to reference point (e.g., wrist).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            Normalized landmarks
        """
        normalized = []
        
        for hand in landmarks:
            if len(hand['landmarks']) <= self.reference_point_idx:
                normalized.append(hand)
                continue
            
            ref_point = hand['landmarks'][self.reference_point_idx]
            ref_x, ref_y, ref_z = ref_point['x'], ref_point['y'], ref_point['z']
            
            normalized_landmarks = []
            for lm in hand['landmarks']:
                normalized_landmarks.append({
                    'x': lm['x'] - ref_x,
                    'y': lm['y'] - ref_y,
                    'z': lm['z'] - ref_z,
                    'visibility': lm.get('visibility', 1.0)
                })
            
            normalized.append({
                'type': hand['type'],
                'landmarks': normalized_landmarks
            })
        
        return normalized


if __name__ == "__main__":
    print("Testing Noise Filtering...")
    
    # Test moving average filter
    filter = LandmarkFilter(window_size=3)
    
    # Create noisy mock data
    for i in range(10):
        noise = np.random.randn() * 0.01
        mock_landmarks = [{
            'type': 'Right',
            'landmarks': [{'x': 0.5 + noise, 'y': 0.5 + noise, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
        }]
        
        smoothed = filter.moving_average(mock_landmarks)
        print(f"Frame {i}: Original x={mock_landmarks[0]['landmarks'][0]['x']:.4f}, "
              f"Smoothed x={smoothed[0]['landmarks'][0]['x']:.4f}")
    
    print("\nTesting Occlusion Handler...")
    handler = OcclusionHandler(max_interpolation_gap=3)
    
    # Simulate occlusion
    valid_landmarks = [{
        'type': 'Left',
        'landmarks': [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
    }]
    
    # Process valid frame
    result = handler.process_frame(valid_landmarks)
    print(f"Valid frame: {len(result)} hands detected")
    
    # Simulate missing hand
    result = handler.process_frame([])
    print(f"Missing frame (interpolated): {len(result)} hands detected")
    
    print("\nRobustness tests completed successfully!")
