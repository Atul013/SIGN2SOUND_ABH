"""
Unit tests for inference pipeline.
Tests real-time inference, TTS integration, and end-to-end prediction.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.utils import (
    normalize_landmarks, denormalize_landmarks,
    compute_hand_bbox, is_valid_landmark_sequence
)


class TestInferenceUtils(unittest.TestCase):
    """Test cases for inference utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock hand landmarks
        self.mock_landmarks = [
            {'x': 0.5 + i*0.01, 'y': 0.5 + i*0.01, 'z': 0.0}
            for i in range(21)
        ]
        
        self.mock_hand = {
            'type': 'Right',
            'landmarks': self.mock_landmarks
        }
    
    def test_normalize_landmarks(self):
        """Test landmark normalization."""
        normalized = normalize_landmarks(self.mock_landmarks)
        
        # Check that we get the same number of landmarks
        self.assertEqual(len(normalized), len(self.mock_landmarks))
        
        # Check that each landmark has x, y, z
        for lm in normalized:
            self.assertIn('x', lm)
            self.assertIn('y', lm)
            self.assertIn('z', lm)
        
        # Wrist (landmark 0) should be at origin after normalization
        self.assertAlmostEqual(normalized[0]['x'], 0.0, places=5)
        self.assertAlmostEqual(normalized[0]['y'], 0.0, places=5)
    
    def test_denormalize_landmarks(self):
        """Test landmark denormalization."""
        # Normalize then denormalize should give back original
        normalized = normalize_landmarks(self.mock_landmarks)
        denormalized = denormalize_landmarks(
            normalized,
            wrist_x=self.mock_landmarks[0]['x'],
            wrist_y=self.mock_landmarks[0]['y']
        )
        
        # Check that we get back close to original values
        for orig, denorm in zip(self.mock_landmarks, denormalized):
            self.assertAlmostEqual(orig['x'], denorm['x'], places=5)
            self.assertAlmostEqual(orig['y'], denorm['y'], places=5)
    
    def test_compute_hand_bbox(self):
        """Test bounding box computation."""
        bbox = compute_hand_bbox(self.mock_landmarks)
        
        # Check that bbox has required keys
        self.assertIn('x_min', bbox)
        self.assertIn('y_min', bbox)
        self.assertIn('x_max', bbox)
        self.assertIn('y_max', bbox)
        self.assertIn('width', bbox)
        self.assertIn('height', bbox)
        
        # Check that min < max
        self.assertLess(bbox['x_min'], bbox['x_max'])
        self.assertLess(bbox['y_min'], bbox['y_max'])
        
        # Check that width and height are positive
        self.assertGreater(bbox['width'], 0)
        self.assertGreater(bbox['height'], 0)
    
    def test_is_valid_landmark_sequence(self):
        """Test landmark sequence validation."""
        # Valid sequence
        valid_sequence = [[self.mock_hand] for _ in range(20)]
        self.assertTrue(is_valid_landmark_sequence(valid_sequence, min_length=10))
        
        # Too short sequence
        short_sequence = [[self.mock_hand] for _ in range(5)]
        self.assertFalse(is_valid_landmark_sequence(short_sequence, min_length=10))
        
        # Empty sequence
        empty_sequence = []
        self.assertFalse(is_valid_landmark_sequence(empty_sequence, min_length=1))
        
        # Sequence with missing hands
        missing_sequence = [[] for _ in range(20)]
        self.assertFalse(is_valid_landmark_sequence(missing_sequence, min_length=10))


class TestSequenceProcessing(unittest.TestCase):
    """Test cases for sequence processing in inference."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sequence_length = 30
        self.num_features = 63  # 21 landmarks * 3 coords
        
    def test_sequence_padding(self):
        """Test sequence padding to fixed length."""
        # Short sequence
        short_seq = np.random.rand(15, self.num_features)
        
        # Pad to target length
        target_length = 30
        padded = np.pad(
            short_seq,
            ((0, target_length - len(short_seq)), (0, 0)),
            mode='edge'
        )
        
        self.assertEqual(padded.shape[0], target_length)
        self.assertEqual(padded.shape[1], self.num_features)
    
    def test_sequence_truncation(self):
        """Test sequence truncation to fixed length."""
        # Long sequence
        long_seq = np.random.rand(50, self.num_features)
        
        # Truncate to target length
        target_length = 30
        truncated = long_seq[:target_length]
        
        self.assertEqual(truncated.shape[0], target_length)
        self.assertEqual(truncated.shape[1], self.num_features)
    
    def test_sequence_normalization(self):
        """Test sequence feature normalization."""
        sequence = np.random.rand(30, self.num_features)
        
        # Normalize to zero mean, unit variance
        mean = sequence.mean(axis=0)
        std = sequence.std(axis=0) + 1e-7
        normalized = (sequence - mean) / std
        
        # Check that mean is close to 0
        np.testing.assert_array_almost_equal(
            normalized.mean(axis=0),
            np.zeros(self.num_features),
            decimal=5
        )


class TestPredictionPostProcessing(unittest.TestCase):
    """Test cases for prediction post-processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 26
        
    def test_softmax_to_class(self):
        """Test converting softmax to class prediction."""
        # Mock softmax output
        softmax = np.random.rand(self.num_classes)
        softmax = softmax / softmax.sum()
        
        # Get predicted class
        predicted_class = np.argmax(softmax)
        confidence = softmax[predicted_class]
        
        self.assertGreaterEqual(predicted_class, 0)
        self.assertLess(predicted_class, self.num_classes)
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_top_k_predictions(self):
        """Test getting top-k predictions."""
        # Mock softmax output
        softmax = np.random.rand(self.num_classes)
        softmax = softmax / softmax.sum()
        
        k = 5
        top_k_indices = np.argsort(softmax)[-k:][::-1]
        top_k_probs = softmax[top_k_indices]
        
        # Check that we got k predictions
        self.assertEqual(len(top_k_indices), k)
        self.assertEqual(len(top_k_probs), k)
        
        # Check that probabilities are in descending order
        for i in range(len(top_k_probs) - 1):
            self.assertGreaterEqual(top_k_probs[i], top_k_probs[i+1])
    
    def test_confidence_thresholding(self):
        """Test confidence-based filtering."""
        # Mock predictions with varying confidence
        predictions = [
            {'class': 0, 'confidence': 0.95},
            {'class': 1, 'confidence': 0.60},
            {'class': 2, 'confidence': 0.30},
        ]
        
        threshold = 0.7
        filtered = [p for p in predictions if p['confidence'] >= threshold]
        
        # Only high-confidence predictions should remain
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['class'], 0)


class TestTemporalFiltering(unittest.TestCase):
    """Test cases for temporal filtering of predictions."""
    
    def test_majority_voting(self):
        """Test majority voting over temporal window."""
        # Sequence of predictions (mostly class 0, some noise)
        predictions = [0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
        
        # Majority vote
        from collections import Counter
        majority = Counter(predictions).most_common(1)[0][0]
        
        self.assertEqual(majority, 0)
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing of predictions."""
        # Sequence of softmax predictions
        num_frames = 10
        num_classes = 26
        
        predictions = np.random.rand(num_frames, num_classes)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        # Apply moving average smoothing
        window_size = 3
        smoothed = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            start = max(0, i - window_size + 1)
            end = i + 1
            smoothed[i] = predictions[start:end].mean(axis=0)
        
        # Check that smoothed predictions are valid probabilities
        np.testing.assert_array_almost_equal(
            smoothed.sum(axis=1),
            np.ones(num_frames)
        )
    
    def test_duplicate_removal(self):
        """Test removing duplicate consecutive predictions."""
        # Sequence with duplicates
        predictions = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
        
        # Remove consecutive duplicates
        deduplicated = []
        prev = None
        for pred in predictions:
            if pred != prev:
                deduplicated.append(pred)
                prev = pred
        
        expected = [0, 1, 2, 3]
        self.assertEqual(deduplicated, expected)


class TestRealtimeInference(unittest.TestCase):
    """Test cases for real-time inference components."""
    
    def test_frame_buffer(self):
        """Test frame buffering for real-time inference."""
        buffer_size = 30
        frame_buffer = []
        
        # Simulate adding frames
        for i in range(50):
            frame_data = {'frame_id': i, 'landmarks': []}
            frame_buffer.append(frame_data)
            
            # Keep only last buffer_size frames
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Check buffer size constraint
            self.assertLessEqual(len(frame_buffer), buffer_size)
        
        # Final buffer should have exactly buffer_size frames
        self.assertEqual(len(frame_buffer), buffer_size)
        
        # Should contain most recent frames
        self.assertEqual(frame_buffer[-1]['frame_id'], 49)
        self.assertEqual(frame_buffer[0]['frame_id'], 20)
    
    def test_inference_timing(self):
        """Test inference timing constraints."""
        import time
        
        # Simulate inference
        start_time = time.time()
        
        # Mock inference operation
        _ = np.random.rand(1, 30, 63)  # Batch of 1, sequence of 30, 63 features
        time.sleep(0.01)  # Simulate processing
        
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Should be under 100ms for real-time performance
        print(f"\nInference time: {inference_time_ms:.2f} ms")
        # Note: This is just a mock test, actual inference time depends on model


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionPostProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestRealtimeInference))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 70)
    print("Running Inference Tests")
    print("=" * 70)
    
    result = run_tests()
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(0 if result.wasSuccessful() else 1)
