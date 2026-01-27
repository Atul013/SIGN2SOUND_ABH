import numpy as np
from typing import List, Tuple, Dict, Optional

class TemporalSegmenter:
    """
    Segments continuous landmark sequences into individual signs based on motion analysis.
    """
    
    def __init__(self, 
                 motion_threshold: float = 0.02,
                 min_sign_duration: int = 10,
                 pause_duration: int = 5,
                 pause_threshold: float = 0.005):
        """
        Initialize the temporal segmenter.
        
        Args:
            motion_threshold: Minimum velocity to consider as motion
            min_sign_duration: Minimum number of frames for a valid sign
            pause_duration: Number of low-motion frames to consider a pause
            pause_threshold: Maximum velocity to consider as pause/rest
        """
        self.motion_threshold = motion_threshold
        self.min_sign_duration = min_sign_duration
        self.pause_duration = pause_duration
        self.pause_threshold = pause_threshold
    
    def compute_velocity(self, landmarks_sequence: List[Dict]) -> np.ndarray:
        """
        Compute frame-to-frame velocity from landmark sequences.
        
        Args:
            landmarks_sequence: List of landmark dicts, each containing hand landmarks
            
        Returns:
            Array of velocity magnitudes for each frame
        """
        if len(landmarks_sequence) < 2:
            return np.array([0.0])
        
        velocities = []
        
        for i in range(1, len(landmarks_sequence)):
            prev_frame = landmarks_sequence[i-1]
            curr_frame = landmarks_sequence[i]
            
            # Calculate velocity for each hand present in both frames
            frame_velocity = 0.0
            hand_count = 0
            
            # Match hands by type (Left/Right)
            for curr_hand in curr_frame:
                # Find matching hand in previous frame
                prev_hand = None
                for ph in prev_frame:
                    if ph['type'] == curr_hand['type']:
                        prev_hand = ph
                        break
                
                if prev_hand:
                    # Compute displacement for all landmarks
                    displacement = 0.0
                    for j, (curr_lm, prev_lm) in enumerate(zip(curr_hand['landmarks'], prev_hand['landmarks'])):
                        dx = curr_lm['x'] - prev_lm['x']
                        dy = curr_lm['y'] - prev_lm['y']
                        dz = curr_lm['z'] - prev_lm['z']
                        displacement += np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Average displacement across all landmarks
                    frame_velocity += displacement / len(curr_hand['landmarks'])
                    hand_count += 1
            
            # Average velocity across all hands
            if hand_count > 0:
                velocities.append(frame_velocity / hand_count)
            else:
                velocities.append(0.0)
        
        return np.array(velocities)
    
    def detect_motion_boundaries(self, velocities: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect sign start and end boundaries based on motion analysis.
        
        Args:
            velocities: Array of frame velocities
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        if len(velocities) == 0:
            return []
        
        segments = []
        in_motion = False
        start_frame = 0
        pause_counter = 0
        
        for i, vel in enumerate(velocities):
            if vel > self.motion_threshold:
                if not in_motion:
                    # Motion started
                    start_frame = i
                    in_motion = True
                    pause_counter = 0
                else:
                    # Continue motion, reset pause counter
                    pause_counter = 0
            else:
                if in_motion:
                    pause_counter += 1
                    
                    # Check if pause is long enough to end the sign
                    if pause_counter >= self.pause_duration:
                        end_frame = i - self.pause_duration
                        
                        # Validate minimum duration
                        if (end_frame - start_frame) >= self.min_sign_duration:
                            segments.append((start_frame, end_frame))
                        
                        in_motion = False
                        pause_counter = 0
        
        # Handle case where motion continues until end
        if in_motion and (len(velocities) - start_frame) >= self.min_sign_duration:
            segments.append((start_frame, len(velocities)))
        
        return segments
    
    def segment_sequence(self, landmarks_sequence: List[Dict]) -> List[Dict]:
        """
        Segment a continuous landmark sequence into individual signs.
        
        Args:
            landmarks_sequence: List of landmark frames
            
        Returns:
            List of segmented sign dicts with 'start', 'end', and 'landmarks' keys
        """
        velocities = self.compute_velocity(landmarks_sequence)
        boundaries = self.detect_motion_boundaries(velocities)
        
        segmented_signs = []
        for start, end in boundaries:
            segmented_signs.append({
                'start_frame': start,
                'end_frame': end,
                'duration': end - start,
                'landmarks': landmarks_sequence[start:end+1]
            })
        
        return segmented_signs
    
    def is_pause(self, landmarks_sequence: List[Dict], window_size: int = 5) -> bool:
        """
        Determine if the current sequence represents a pause/rest state.
        
        Args:
            landmarks_sequence: Recent landmark frames
            window_size: Number of frames to analyze
            
        Returns:
            True if sequence represents a pause
        """
        if len(landmarks_sequence) < window_size:
            return False
        
        recent_frames = landmarks_sequence[-window_size:]
        velocities = self.compute_velocity(recent_frames)
        
        # Check if all recent velocities are below pause threshold
        return np.all(velocities < self.pause_threshold)


if __name__ == "__main__":
    # Simple test with mock data
    print("Testing Temporal Segmentation...")
    
    # Create mock landmark sequence (simulating hand movement)
    mock_sequence = []
    
    # Rest phase (frames 0-9)
    for i in range(10):
        mock_sequence.append([{
            'type': 'Right',
            'landmarks': [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(21)]
        }])
    
    # Motion phase (frames 10-29)
    for i in range(20):
        t = i / 20.0
        mock_sequence.append([{
            'type': 'Right',
            'landmarks': [{'x': 0.5 + t * 0.1, 'y': 0.5 + t * 0.1, 'z': 0.0} for _ in range(21)]
        }])
    
    # Rest phase (frames 30-39)
    for i in range(10):
        mock_sequence.append([{
            'type': 'Right',
            'landmarks': [{'x': 0.6, 'y': 0.6, 'z': 0.0} for _ in range(21)]
        }])
    
    segmenter = TemporalSegmenter()
    segments = segmenter.segment_sequence(mock_sequence)
    
    print(f"\nDetected {len(segments)} sign(s):")
    for idx, seg in enumerate(segments):
        print(f"  Sign {idx+1}: frames {seg['start_frame']}-{seg['end_frame']} (duration: {seg['duration']})")
    
    # Test pause detection
    is_paused = segmenter.is_pause(mock_sequence[-10:])
    print(f"\nLast 10 frames represent pause: {is_paused}")
