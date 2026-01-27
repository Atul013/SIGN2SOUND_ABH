from collections import deque
from typing import List, Dict, Optional
import numpy as np

class SequenceBuffer:
    """
    Manages a sliding window buffer for continuous landmark input.
    Handles real-time streaming and overlapping sequences.
    """
    
    def __init__(self, 
                 max_buffer_size: int = 100,
                 overlap_frames: int = 10):
        """
        Initialize the sequence buffer.
        
        Args:
            max_buffer_size: Maximum number of frames to keep in buffer
            overlap_frames: Number of frames to overlap between segments
        """
        self.max_buffer_size = max_buffer_size
        self.overlap_frames = overlap_frames
        self.buffer = deque(maxlen=max_buffer_size)
        self.frame_count = 0
    
    def add_frame(self, landmarks: List[Dict]) -> None:
        """
        Add a new frame of landmarks to the buffer.
        
        Args:
            landmarks: List of hand landmark dicts for this frame
        """
        self.buffer.append({
            'frame_id': self.frame_count,
            'landmarks': landmarks,
            'timestamp': self.frame_count  # Could be replaced with actual timestamp
        })
        self.frame_count += 1
    
    def get_buffer(self) -> List[Dict]:
        """
        Get the current buffer contents.
        
        Returns:
            List of buffered frames
        """
        return list(self.buffer)
    
    def get_recent_frames(self, n: int) -> List[Dict]:
        """
        Get the n most recent frames.
        
        Args:
            n: Number of recent frames to retrieve
            
        Returns:
            List of recent frames (may be less than n if buffer is smaller)
        """
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.frame_count = 0
    
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return len(self.buffer) >= self.max_buffer_size
    
    def get_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def extract_landmarks_only(self) -> List[List[Dict]]:
        """
        Extract just the landmark data from buffered frames.
        
        Returns:
            List of landmark lists (one per frame)
        """
        return [frame['landmarks'] for frame in self.buffer]
    
    def create_overlapping_windows(self, window_size: int) -> List[List[Dict]]:
        """
        Create overlapping windows from the buffer for batch processing.
        
        Args:
            window_size: Size of each window
            
        Returns:
            List of overlapping windows
        """
        if len(self.buffer) < window_size:
            return []
        
        windows = []
        step_size = window_size - self.overlap_frames
        
        for i in range(0, len(self.buffer) - window_size + 1, step_size):
            window = list(self.buffer)[i:i + window_size]
            windows.append([frame['landmarks'] for frame in window])
        
        return windows


class RealTimeProcessor:
    """
    Processes landmarks in real-time with buffering and segmentation.
    """
    
    def __init__(self, 
                 buffer_size: int = 100,
                 segment_callback: Optional[callable] = None):
        """
        Initialize the real-time processor.
        
        Args:
            buffer_size: Size of the frame buffer
            segment_callback: Function to call when a segment is detected
        """
        self.buffer = SequenceBuffer(max_buffer_size=buffer_size)
        self.segment_callback = segment_callback
        self.last_processed_frame = 0
    
    def process_frame(self, landmarks: List[Dict]) -> Optional[Dict]:
        """
        Process a single frame of landmarks.
        
        Args:
            landmarks: Hand landmarks for this frame
            
        Returns:
            Segment dict if a complete sign was detected, None otherwise
        """
        self.buffer.add_frame(landmarks)
        
        # Check if we have enough frames to analyze
        if self.buffer.get_size() < 20:  # Minimum frames needed
            return None
        
        # This is a placeholder - in practice, you'd integrate with TemporalSegmenter
        # to detect when a sign is complete and extract it
        
        return None
    
    def get_current_buffer(self) -> List[List[Dict]]:
        """Get current buffer state."""
        return self.buffer.extract_landmarks_only()
    
    def reset(self) -> None:
        """Reset the processor state."""
        self.buffer.clear()
        self.last_processed_frame = 0


if __name__ == "__main__":
    print("Testing Sequence Buffer...")
    
    # Test basic buffering
    buffer = SequenceBuffer(max_buffer_size=50, overlap_frames=5)
    
    # Add some mock frames
    for i in range(30):
        mock_landmarks = [{
            'type': 'Right',
            'landmarks': [{'x': i * 0.01, 'y': 0.5, 'z': 0.0} for _ in range(21)]
        }]
        buffer.add_frame(mock_landmarks)
    
    print(f"Buffer size: {buffer.get_size()}")
    print(f"Buffer full: {buffer.is_full()}")
    
    # Test overlapping windows
    windows = buffer.create_overlapping_windows(window_size=10)
    print(f"Created {len(windows)} overlapping windows")
    
    # Test recent frames
    recent = buffer.get_recent_frames(5)
    print(f"Retrieved {len(recent)} recent frames")
    
    # Test real-time processor
    print("\nTesting Real-Time Processor...")
    processor = RealTimeProcessor(buffer_size=50)
    
    for i in range(25):
        mock_landmarks = [{
            'type': 'Left',
            'landmarks': [{'x': 0.5, 'y': i * 0.02, 'z': 0.0} for _ in range(21)]
        }]
        result = processor.process_frame(mock_landmarks)
    
    current_buffer = processor.get_current_buffer()
    print(f"Processor buffer contains {len(current_buffer)} frames")
    
    print("\nSequence buffering tests completed successfully!")
