"""
Inference utilities for real-time sign language recognition.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.hand_landmarks import HandLandmarks
from features.feature_utils import FeatureContract, FeatureNormalizer
from preprocessing.sequence_buffer import SequenceBuffer
from preprocessing.temporal_segmentation import TemporalSegmenter
from preprocessing.robustness import LandmarkFilter


class RealTimeInference:
    """
    Real-time inference engine for sign language recognition.
    """
    
    def __init__(self,
                 hand_model_path: str = 'models/hand_landmarker.task',
                 buffer_size: int = 100,
                 enable_filtering: bool = True):
        """
        Initialize real-time inference.
        
        Args:
            hand_model_path: Path to hand landmarker model
            buffer_size: Size of frame buffer
            enable_filtering: Whether to apply noise filtering
        """
        self.hand_detector = HandLandmarks(model_path=hand_model_path)
        self.buffer = SequenceBuffer(max_buffer_size=buffer_size)
        self.segmenter = TemporalSegmenter()
        self.filter = LandmarkFilter() if enable_filtering else None
        
        self.frame_count = 0
        self.last_segment_frame = 0
    
    def process_frame(self, image: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame in real-time.
        
        Args:
            image: BGR image from camera
            
        Returns:
            Detected sign segment or None
        """
        # Extract landmarks
        _, hand_data = self.hand_detector.extract_landmarks(image)
        
        # Apply filtering
        if self.filter and hand_data:
            hand_data = self.filter.moving_average(hand_data)
        
        # Add to buffer
        self.buffer.add_frame(hand_data)
        self.frame_count += 1
        
        # Check if we can detect a sign
        if self.buffer.get_size() < 20:
            return None
        
        # Get recent frames for analysis
        recent_frames = self.buffer.get_recent_frames(50)
        hand_sequence = [frame['landmarks'] for frame in recent_frames]
        
        # Check for pause (sign completion)
        if self.segmenter.is_pause(hand_sequence):
            # Try to extract the last sign
            segments = self.segmenter.segment_sequence(hand_sequence)
            
            if segments:
                # Return the most recent complete segment
                last_segment = segments[-1]
                
                # Avoid returning the same segment twice
                if last_segment['end_frame'] > self.last_segment_frame:
                    self.last_segment_frame = last_segment['end_frame']
                    return {
                        'landmarks': last_segment['landmarks'],
                        'duration': last_segment['duration'],
                        'frame_id': self.frame_count
                    }
        
        return None
    
    def reset(self):
        """Reset inference state."""
        self.buffer.clear()
        self.frame_count = 0
        self.last_segment_frame = 0
        if self.filter:
            self.filter.reset()


class WebcamCapture:
    """
    Utility for capturing from webcam with visualization.
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480):
        """
        Initialize webcam capture.
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from camera."""
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release camera."""
        self.cap.release()
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap.isOpened()


def draw_landmarks_on_frame(image: np.ndarray, 
                            hand_data: List[Dict],
                            show_connections: bool = True) -> np.ndarray:
    """
    Draw hand landmarks on image.
    
    Args:
        image: Image to draw on
        hand_data: List of hand landmark dicts
        show_connections: Whether to draw connections
        
    Returns:
        Image with landmarks drawn
    """
    if not hand_data:
        return image
    
    h, w, _ = image.shape
    
    # Hand connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
        (5, 9), (9, 13), (13, 17), (0, 17)        # Palm
    ]
    
    for hand in hand_data:
        landmarks = hand['landmarks']
        hand_type = hand['type']
        
        # Choose color based on hand type
        color = (0, 255, 0) if hand_type == 'Right' else (255, 0, 0)
        
        # Draw connections
        if show_connections:
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_lm = landmarks[start_idx]
                    end_lm = landmarks[end_idx]
                    
                    start_point = (int(start_lm['x'] * w), int(start_lm['y'] * h))
                    end_point = (int(end_lm['x'] * w), int(end_lm['y'] * h))
                    
                    cv2.line(image, start_point, end_point, color, 2)
        
        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm['x'] * w), int(lm['y'] * h)
            cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1)
            cv2.circle(image, (cx, cy), 2, color, -1)
    
    return image


def add_info_overlay(image: np.ndarray,
                     info_text: List[str],
                     position: tuple = (10, 30)) -> np.ndarray:
    """
    Add text overlay to image.
    
    Args:
        image: Image to draw on
        info_text: List of text lines
        position: Starting position (x, y)
        
    Returns:
        Image with text overlay
    """
    x, y = position
    
    for i, text in enumerate(info_text):
        cv2.putText(image, text, (x, y + i * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time sign language inference')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--buffer-size', type=int, default=100, help='Buffer size')
    
    args = parser.parse_args()
    
    print("Initializing real-time inference...")
    inference = RealTimeInference(buffer_size=args.buffer_size)
    webcam = WebcamCapture(camera_index=args.camera)
    
    print("Starting real-time capture...")
    print("Press 'q' to quit")
    
    sign_count = 0
    
    try:
        while webcam.is_opened():
            frame = webcam.read_frame()
            if frame is None:
                break
            
            # Process frame
            result = inference.process_frame(frame)
            
            # Get current buffer state for visualization
            recent = inference.buffer.get_recent_frames(1)
            hand_data = recent[0]['landmarks'] if recent else []
            
            # Draw landmarks
            frame = draw_landmarks_on_frame(frame, hand_data)
            
            # Add info overlay
            info = [
                f"Frames: {inference.frame_count}",
                f"Buffer: {inference.buffer.get_size()}/{args.buffer_size}",
                f"Signs detected: {sign_count}"
            ]
            
            if result:
                sign_count += 1
                info.append(f"SIGN DETECTED! (duration: {result['duration']} frames)")
            
            frame = add_info_overlay(frame, info)
            
            # Display
            cv2.imshow('Real-Time Sign Recognition', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        print(f"\nSession complete. Detected {sign_count} signs.")
