"""
Unit tests for feature extraction modules.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_utils import FeatureContract, FeatureSerializer, FeatureNormalizer


class TestFeatureContract(unittest.TestCase):
    """Test feature data contract."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_landmarks = [
            {'x': i * 0.1, 'y': i * 0.1, 'z': 0.0, 'visibility': 1.0}
            for i in range(21)
        ]
    
    def test_create_hand_data(self):
        """Test hand data creation."""
        hand_data = FeatureContract.create_hand_data('Right', self.mock_landmarks)
        
        self.assertEqual(hand_data['type'], 'Right')
        self.assertEqual(len(hand_data['landmarks']), 21)
        self.assertEqual(hand_data['num_landmarks'], 21)
    
    def test_validate_hand_data(self):
        """Test hand data validation."""
        hand_data = FeatureContract.create_hand_data('Left', self.mock_landmarks)
        self.assertTrue(FeatureContract.validate_hand_data(hand_data))
        
        # Test invalid data
        invalid_data = {'type': 'Invalid', 'landmarks': []}
        self.assertFalse(FeatureContract.validate_hand_data(invalid_data))
    
    def test_create_frame_data(self):
        """Test frame data creation."""
        hand_data = FeatureContract.create_hand_data('Right', self.mock_landmarks)
        frame_data = FeatureContract.create_frame_data(0, [hand_data])
        
        self.assertEqual(frame_data['frame_id'], 0)
        self.assertEqual(frame_data['num_hands'], 1)
        self.assertTrue(FeatureContract.validate_frame_data(frame_data))


class TestFeatureSerializer(unittest.TestCase):
    """Test feature serialization."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_landmarks = [
            {'x': i * 0.1, 'y': i * 0.1, 'z': 0.0, 'visibility': 1.0}
            for i in range(21)
        ]
    
    def test_to_numpy(self):
        """Test conversion to numpy."""
        array = FeatureSerializer.to_numpy(self.mock_landmarks)
        
        self.assertEqual(array.shape, (21, 3))
        self.assertAlmostEqual(array[0, 0], 0.0)
        self.assertAlmostEqual(array[10, 0], 1.0)
    
    def test_from_numpy(self):
        """Test conversion from numpy."""
        array = FeatureSerializer.to_numpy(self.mock_landmarks)
        landmarks = FeatureSerializer.from_numpy(array)
        
        self.assertEqual(len(landmarks), 21)
        self.assertAlmostEqual(landmarks[0]['x'], 0.0)
        self.assertAlmostEqual(landmarks[10]['x'], 1.0)
    
    def test_sequence_to_numpy(self):
        """Test sequence conversion."""
        hand_data = FeatureContract.create_hand_data('Right', self.mock_landmarks)
        sequence = [[hand_data], [hand_data]]
        
        array = FeatureSerializer.sequence_to_numpy(sequence)
        
        self.assertEqual(array.shape, (2, 2, 21, 3))  # 2 frames, 2 hands (padded), 21 landmarks, 3 coords


class TestFeatureNormalizer(unittest.TestCase):
    """Test feature normalization."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_landmarks = [
            {'x': 0.5 + i * 0.01, 'y': 0.5 + i * 0.01, 'z': 0.0, 'visibility': 1.0}
            for i in range(21)
        ]
        self.hand_data = FeatureContract.create_hand_data('Right', self.mock_landmarks)
    
    def test_normalize_to_wrist(self):
        """Test wrist normalization."""
        normalized = FeatureNormalizer.normalize_hand_to_wrist(self.hand_data)
        
        # Wrist should be at origin
        self.assertAlmostEqual(normalized['landmarks'][0]['x'], 0.0)
        self.assertAlmostEqual(normalized['landmarks'][0]['y'], 0.0)
        self.assertAlmostEqual(normalized['landmarks'][0]['z'], 0.0)
    
    def test_scale_to_unit_box(self):
        """Test unit box scaling."""
        scaled = FeatureNormalizer.scale_to_unit_box(self.mock_landmarks)
        
        # Check that values are in [0, 1]
        for lm in scaled:
            self.assertGreaterEqual(lm['x'], 0.0)
            self.assertLessEqual(lm['x'], 1.0)
            self.assertGreaterEqual(lm['y'], 0.0)
            self.assertLessEqual(lm['y'], 1.0)


if __name__ == '__main__':
    unittest.main()
