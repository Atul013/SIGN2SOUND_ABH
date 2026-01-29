"""
Unit tests for model architecture and training.
Tests model building, forward pass, and training components.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ASLRecognitionModel, get_recommended_model
from models.custom_layers import (
    AttentionLayer, TemporalAttention, MultiHeadAttention,
    PositionalEncoding, get_custom_layer_registry
)
from models.loss import (
    FocalLoss, TemporalConsistencyLoss, LabelSmoothingLoss,
    compute_class_weights, get_loss_registry
)


class TestASLRecognitionModel(unittest.TestCase):
    """Test cases for ASL recognition model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 63  # 21 landmarks * 3 coordinates
        self.num_classes = 26  # A-Z
        
    def test_model_initialization(self):
        """Test model initialization with different types."""
        for model_type in ['lstm', 'gru', 'cnn']:
            model = ASLRecognitionModel(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                model_type=model_type
            )
            
            self.assertEqual(model.input_dim, self.input_dim)
            self.assertEqual(model.num_classes, self.num_classes)
            self.assertEqual(model.model_type, model_type)
    
    def test_lstm_model_architecture(self):
        """Test LSTM model architecture building."""
        model = ASLRecognitionModel(model_type='lstm')
        architecture = model.build_lstm_model(
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        
        self.assertEqual(architecture['type'], 'LSTM')
        self.assertIn('layers', architecture)
        self.assertIn('input_shape', architecture)
        self.assertIn('output_shape', architecture)
        
        # Check output shape matches num_classes
        self.assertEqual(architecture['output_shape'], (self.num_classes,))
    
    def test_gru_model_architecture(self):
        """Test GRU model architecture building."""
        model = ASLRecognitionModel(model_type='gru')
        architecture = model.build_gru_model(
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        
        self.assertEqual(architecture['type'], 'GRU')
        self.assertIn('layers', architecture)
    
    def test_cnn_model_architecture(self):
        """Test CNN model architecture building."""
        model = ASLRecognitionModel(model_type='cnn')
        architecture = model.build_cnn_model(
            conv_filters=[64, 128, 256],
            kernel_size=3,
            dropout=0.3
        )
        
        self.assertEqual(architecture['type'], 'CNN')
        self.assertIn('layers', architecture)
    
    def test_parameter_estimation(self):
        """Test parameter count estimation."""
        model = ASLRecognitionModel(model_type='lstm')
        architecture = model.build_lstm_model(hidden_size=64, num_layers=1)
        
        params = model._estimate_parameters(architecture)
        
        # Should have positive number of parameters
        self.assertGreater(params, 0)
        
        # Model should be under 10MB for edge deployment
        size_mb = params * 4 / (1024 * 1024)
        print(f"\nEstimated model size: {size_mb:.2f} MB")
    
    def test_get_recommended_model(self):
        """Test recommended model configurations."""
        use_cases = ['static', 'sequence', 'hybrid']
        
        for use_case in use_cases:
            model_type, params = get_recommended_model(use_case)
            
            self.assertIsInstance(model_type, str)
            self.assertIsInstance(params, dict)
            self.assertIn(model_type, ['lstm', 'gru', 'cnn'])
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        model = ASLRecognitionModel(model_type='invalid')
        
        with self.assertRaises(ValueError):
            model.build()


class TestCustomLayers(unittest.TestCase):
    """Test cases for custom layers."""
    
    def test_attention_layer(self):
        """Test attention layer configuration."""
        layer = AttentionLayer(units=128, return_attention_weights=True)
        config = layer.get_config()
        
        self.assertEqual(config['type'], 'Attention')
        self.assertEqual(config['units'], 128)
        self.assertTrue(config['return_attention_weights'])
    
    def test_temporal_attention(self):
        """Test temporal attention layer."""
        layer = TemporalAttention(units=128)
        config = layer.get_config()
        
        self.assertEqual(config['type'], 'TemporalAttention')
        self.assertEqual(config['units'], 128)
        self.assertIn('architecture', config)
    
    def test_multi_head_attention(self):
        """Test multi-head attention layer."""
        layer = MultiHeadAttention(num_heads=4, key_dim=64)
        config = layer.get_config()
        
        self.assertEqual(config['type'], 'MultiHeadAttention')
        self.assertEqual(config['num_heads'], 4)
        self.assertEqual(config['key_dim'], 64)
        self.assertEqual(config['total_dim'], 256)  # 4 * 64
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        layer = PositionalEncoding(max_sequence_length=100, embedding_dim=128)
        config = layer.get_config()
        
        self.assertEqual(config['type'], 'PositionalEncoding')
        self.assertEqual(config['max_sequence_length'], 100)
        self.assertEqual(config['embedding_dim'], 128)
        
        # Test positional encoding computation
        pe = PositionalEncoding.compute_positional_encoding(100, 128)
        self.assertEqual(pe.shape, (100, 128))
    
    def test_custom_layer_registry(self):
        """Test custom layer registry."""
        registry = get_custom_layer_registry()
        
        self.assertIsInstance(registry, dict)
        self.assertGreater(len(registry), 0)
        
        # Check that all registered layers are classes
        for name, layer_class in registry.items():
            self.assertTrue(callable(layer_class))


class TestLossFunctions(unittest.TestCase):
    """Test cases for custom loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 26
        self.batch_size = 32
        
        # Create mock data
        self.y_true = np.eye(self.num_classes)[np.random.randint(0, self.num_classes, self.batch_size)]
        self.y_pred = np.random.rand(self.batch_size, self.num_classes)
        self.y_pred = self.y_pred / self.y_pred.sum(axis=1, keepdims=True)  # Normalize
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        config = loss.get_config()
        
        self.assertEqual(config['type'], 'FocalLoss')
        self.assertEqual(config['alpha'], 0.25)
        self.assertEqual(config['gamma'], 2.0)
        
        # Compute loss
        loss_value = loss.compute(self.y_true, self.y_pred)
        self.assertGreater(loss_value, 0)
    
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        loss = TemporalConsistencyLoss(weight=0.1)
        config = loss.get_config()
        
        self.assertEqual(config['type'], 'TemporalConsistencyLoss')
        self.assertEqual(config['weight'], 0.1)
        
        # Create sequence of predictions
        seq_len = 20
        predictions_seq = np.random.rand(seq_len, self.num_classes)
        
        loss_value = loss.compute(predictions_seq)
        self.assertGreaterEqual(loss_value, 0)
    
    def test_label_smoothing_loss(self):
        """Test label smoothing."""
        loss = LabelSmoothingLoss(smoothing=0.1)
        config = loss.get_config()
        
        self.assertEqual(config['type'], 'LabelSmoothingLoss')
        self.assertEqual(config['smoothing'], 0.1)
        
        # Test label smoothing
        smoothed = loss.smooth_labels(self.y_true, self.num_classes)
        
        # Check that smoothed labels sum to 1
        np.testing.assert_array_almost_equal(
            smoothed.sum(axis=1),
            np.ones(self.batch_size)
        )
        
        # Check that max value is less than 1 (due to smoothing)
        self.assertLess(smoothed.max(), 1.0)
    
    def test_class_weights_computation(self):
        """Test class weight computation."""
        class_counts = {
            0: 1000,
            1: 500,
            2: 100,
            3: 50
        }
        
        for method in ['balanced', 'inverse', 'sqrt_inverse']:
            weights = compute_class_weights(class_counts, method=method)
            
            # Check that all classes have weights
            self.assertEqual(len(weights), len(class_counts))
            
            # Check that rare classes have higher weights
            self.assertGreater(weights[3], weights[0])
            
            # Check that weights are positive
            for weight in weights.values():
                self.assertGreater(weight, 0)
    
    def test_loss_registry(self):
        """Test loss function registry."""
        registry = get_loss_registry()
        
        self.assertIsInstance(registry, dict)
        self.assertGreater(len(registry), 0)
        
        # Check that all registered losses are classes
        for name, loss_class in registry.items():
            self.assertTrue(callable(loss_class))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def test_model_with_custom_layers(self):
        """Test that model can be extended with custom layers."""
        model = ASLRecognitionModel(model_type='lstm')
        architecture = model.build_lstm_model()
        
        # Custom layers can be added to architecture
        attention = TemporalAttention(units=128)
        
        # Both should have compatible configurations
        self.assertIn('type', architecture)
        self.assertIn('type', attention.get_config())
    
    def test_model_size_constraints(self):
        """Test that models meet size constraints for edge deployment."""
        # Test lightweight configurations
        configs = [
            ('lstm', {'hidden_size': 64, 'num_layers': 1}),
            ('gru', {'hidden_size': 64, 'num_layers': 1}),
            ('cnn', {'conv_filters': [32, 64], 'kernel_size': 3})
        ]
        
        for model_type, params in configs:
            model = ASLRecognitionModel(model_type=model_type)
            architecture = model.build(**params)
            
            param_count = model._estimate_parameters(architecture)
            size_mb = param_count * 4 / (1024 * 1024)
            
            # Should be under 10MB for edge deployment
            print(f"\n{model_type.upper()} model size: {size_mb:.2f} MB")
            self.assertLess(size_mb, 10.0, 
                          f"{model_type} model exceeds 10MB limit: {size_mb:.2f} MB")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestASLRecognitionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 70)
    print("Running Model Tests")
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
