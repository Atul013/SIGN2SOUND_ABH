"""
Unit tests for preprocessing modules.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.temporal_segmentation import TemporalSegmenter
from preprocessing.sequence_buffer import SequenceBuffer
from preprocessing.robustness import LandmarkFilter, OcclusionHandler


class TestTemporalSegmenter(unittest.TestCase):
    """Test temporal segmentation."""
    
    def setUp(self):
        """Set up test data."""
        self.segmenter = TemporalSegmenter(
            motion_threshold=0.02,
            min_sign_duration=5
        )
    
    def create_mock_sequence(self, num_frames: int, motion: bool = True) -> list:
        """Create mock landmark sequence."""
        sequence = []
        for i in range(num_frames):
            if motion:
                x = 0.5 + (i / num_frames) * 0.2
            else:
                x = 0.5
            
            sequence.append([{
                'type': 'Right',
                'landmarks': [{'x': x, 'y': 0.5, 'z': 0.0} for _ in range(21)]
            }])
        
        return sequence
    
    def test_compute_velocity(self):
        """Test velocity computation."""
        sequence = self.create_mock_sequence(20, motion=True)
        velocities = self.segmenter.compute_velocity(sequence)
        
        self.assertEqual(len(velocities), 19)  # n-1 velocities for n frames
        self.assertGreater(np.mean(velocities), 0)  # Should have motion
    
    def test_detect_motion_boundaries(self):
        """Test motion boundary detection."""
        # Create sequence with motion in middle
        sequence = []
        sequence.extend(self.create_mock_sequence(10, motion=False))  # Rest
        sequence.extend(self.create_mock_sequence(15, motion=True))   # Motion
        sequence.extend(self.create_mock_sequence(10, motion=False))  # Rest
        
        velocities = self.segmenter.compute_velocity(sequence)
        boundaries = self.segmenter.detect_motion_boundaries(velocities)
        
        self.assertGreater(len(boundaries), 0)  # Should detect at least one segment
    
    def test_is_pause(self):
        """Test pause detection."""
        rest_sequence = self.create_mock_sequence(10, motion=False)
        motion_sequence = self.create_mock_sequence(10, motion=True)
        
        self.assertTrue(self.segmenter.is_pause(rest_sequence))
        self.assertFalse(self.segmenter.is_pause(motion_sequence))


class TestSequenceBuffer(unittest.TestCase):
    """Test sequence buffering."""
    
    def setUp(self):
        """Set up test buffer."""
        self.buffer = SequenceBuffer(max_buffer_size=50, overlap_frames=5)
    
    def test_add_frame(self):
        """Test adding frames."""
        mock_landmarks = [{'type': 'Right', 'landmarks': []}]
        
        self.buffer.add_frame(mock_landmarks)
        self.assertEqual(self.buffer.get_size(), 1)
        
        self.buffer.add_frame(mock_landmarks)
        self.assertEqual(self.buffer.get_size(), 2)
    
    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        mock_landmarks = [{'type': 'Right', 'landmarks': []}]
        
        # Add more than max size
        for _ in range(60):
            self.buffer.add_frame(mock_landmarks)
        
        # Should be capped at max size
        self.assertEqual(self.buffer.get_size(), 50)
    
    def test_get_recent_frames(self):
        """Test getting recent frames."""
        mock_landmarks = [{'type': 'Right', 'landmarks': []}]
        
        for i in range(20):
            self.buffer.add_frame(mock_landmarks)
        
        recent = self.buffer.get_recent_frames(5)
        self.assertEqual(len(recent), 5)
    
    def test_overlapping_windows(self):
        """Test overlapping window creation."""
        mock_landmarks = [{'type': 'Right', 'landmarks': []}]
        
        for _ in range(30):
            self.buffer.add_frame(mock_landmarks)
        
        windows = self.buffer.create_overlapping_windows(window_size=10)
        self.assertGreater(len(windows), 0)


class TestLandmarkFilter(unittest.TestCase):
    """Test landmark filtering."""
    
    def setUp(self):
        """Set up test filter."""
        self.filter = LandmarkFilter(window_size=3)
    
    def test_moving_average(self):
        """Test moving average filter."""
        # Create noisy data
        for i in range(5):
            noise = np.random.randn() * 0.01
            mock_landmarks = [{
                'type': 'Right',
                'landmarks': [{'x': 0.5 + noise, 'y': 0.5, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
            }]
            
            smoothed = self.filter.moving_average(mock_landmarks)
            self.assertEqual(len(smoothed), 1)
            self.assertEqual(len(smoothed[0]['landmarks']), 21)


class TestOcclusionHandler(unittest.TestCase):
    """Test occlusion handling."""
    
    def setUp(self):
        """Set up test handler."""
        self.handler = OcclusionHandler(max_interpolation_gap=3)
    
    def test_valid_landmarks(self):
        """Test processing valid landmarks."""
        valid_landmarks = [{
            'type': 'Right',
            'landmarks': [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
        }]
        
        result = self.handler.process_frame(valid_landmarks)
        self.assertEqual(len(result), 1)
    
    def test_interpolation(self):
        """Test interpolation of missing data."""
        # First, process valid frame
        valid_landmarks = [{
            'type': 'Right',
            'landmarks': [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
        }]
        self.handler.process_frame(valid_landmarks)
        
        # Then process empty frame (should interpolate)
        result = self.handler.process_frame([])
        self.assertEqual(len(result), 1)  # Should have interpolated hand


if __name__ == '__main__':
    unittest.main()
