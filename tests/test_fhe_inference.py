"""
Test suite for FHE ML inference system.

Tests cover:
- FHE context and cryptographic operations
- Encryption/decryption correctness
- Model predictions
- Activation approximations
- Client-server compatibility
"""

import sys
sys.path.insert(0, '/Users/lokeshjoshi/Documents/Code Bases/phe-interface')

import unittest
import numpy as np
from src.fhe.context import FHEContext
from src.ml.models import LogisticRegression, SimpleNeuralNetwork
from src.ml.activations import (
    SigmoidApproximation,
    ReLUApproximation,
    TanhApproximation,
    ActivationFactory,
)


class TestFHEContext(unittest.TestCase):
    """Test FHE context initialization and operations."""
    
    def setUp(self):
        """Set up test context."""
        self.context = FHEContext(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[60, 40, 40, 60],
            scale_bits=40,
        )
    
    def test_context_initialization(self):
        """Test context creation."""
        self.assertIsNotNone(self.context)
        self.assertEqual(self.context.poly_modulus_degree, 8192)
        self.assertEqual(self.context.scale_bits, 40)
    
    def test_get_parameters_summary(self):
        """Test parameter summary."""
        summary = self.context.get_parameters_summary()
        self.assertIn('poly_modulus_degree', summary)
        self.assertIn('max_multiplicative_depth', summary)
    
    def test_encrypt_decrypt_single_value(self):
        """Test encryption/decryption of single value."""
        plaintext = np.array([0.5], dtype=np.float32)
        encrypted = self.context.encrypt(plaintext)
        decrypted = self.context.decrypt(encrypted)
        
        self.assertEqual(len(decrypted), 1)
        np.testing.assert_almost_equal(decrypted[0], 0.5, decimal=3)
    
    def test_encrypt_decrypt_vector(self):
        """Test encryption/decryption of vectors."""
        plaintext = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        encrypted = self.context.encrypt(plaintext)
        decrypted = self.context.decrypt(encrypted)
        
        self.assertEqual(len(decrypted), 5)
        np.testing.assert_array_almost_equal(decrypted, plaintext, decimal=3)
    
    def test_homomorphic_addition(self):
        """Test E(x) + E(y) = E(x+y)."""
        x = np.array([0.5, 0.3], dtype=np.float32)
        y = np.array([0.2, 0.4], dtype=np.float32)
        
        encrypted_x = self.context.encrypt(x)
        encrypted_y = self.context.encrypt(y)
        
        # Encrypted addition
        encrypted_sum = self.context.add(encrypted_x, encrypted_y)
        decrypted_sum = self.context.decrypt(encrypted_sum)
        
        # Expected
        expected_sum = x + y
        
        np.testing.assert_array_almost_equal(decrypted_sum, expected_sum, decimal=2)
    
    def test_homomorphic_multiplication(self):
        """Test E(x) * E(y) ≈ E(x*y)."""
        x = np.array([0.5], dtype=np.float32)
        y = np.array([0.2], dtype=np.float32)
        
        encrypted_x = self.context.encrypt(x)
        encrypted_y = self.context.encrypt(y)
        
        # Encrypted multiplication
        encrypted_product = self.context.multiply(encrypted_x, encrypted_y)
        decrypted_product = self.context.decrypt(encrypted_product)
        
        # Expected
        expected_product = x * y
        
        # Note: Higher error tolerance for multiplication due to noise accumulation
        np.testing.assert_almost_equal(decrypted_product[0], expected_product[0], decimal=2)
    
    def test_plaintext_multiplication(self):
        """Test E(x) * p = E(x*p)."""
        x = np.array([0.5, 0.3], dtype=np.float32)
        plaintext = np.array([2.0, 3.0], dtype=np.float32)
        
        encrypted_x = self.context.encrypt(x)
        encrypted_result = self.context.multiply_plaintext(encrypted_x, plaintext)
        decrypted_result = self.context.decrypt(encrypted_result)
        
        expected = x * plaintext
        np.testing.assert_array_almost_equal(decrypted_result, expected, decimal=2)


class TestActivationApproximations(unittest.TestCase):
    """Test polynomial activation approximations."""
    
    def test_sigmoid_approximation(self):
        """Test sigmoid polynomial approximation."""
        sigmoid = SigmoidApproximation(degree=3, domain=(-4, 4))
        
        # Test known values
        test_x = np.array([-4, -2, 0, 2, 4], dtype=np.float32)
        approx_y = sigmoid.forward(test_x)
        
        # Sigmoid should be monotonically increasing
        for i in range(len(approx_y) - 1):
            self.assertLess(approx_y[i], approx_y[i+1])
        
        # In the middle, sigmoid should be close to 0.5
        self.assertGreater(approx_y[2], 0.4)
        self.assertLess(approx_y[2], 0.6)
    
    def test_relu_approximation(self):
        """Test ReLU polynomial approximation."""
        relu = ReLUApproximation(degree=3, domain=(-1, 2))
        
        test_x = np.array([-1, -0.5, 0, 0.5, 1, 2], dtype=np.float32)
        approx_y = relu.forward(test_x)
        
        # ReLU should be non-negative
        self.assertTrue(np.all(approx_y >= 0))
        
        # ReLU should be increasing on positive side
        self.assertLess(approx_y[2], approx_y[3])
        self.assertLess(approx_y[3], approx_y[4])
    
    def test_tanh_approximation(self):
        """Test tanh polynomial approximation."""
        tanh = TanhApproximation(degree=3, domain=(-2, 2))
        
        test_x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        approx_y = tanh.forward(test_x)
        
        # Tanh should be monotonically increasing
        for i in range(len(approx_y) - 1):
            self.assertLess(approx_y[i], approx_y[i+1])
        
        # Tanh should be anti-symmetric around 0
        self.assertAlmostEqual(approx_y[2], 0.0, places=1)
    
    def test_activation_factory(self):
        """Test activation factory."""
        sigmoid = ActivationFactory.create('sigmoid', degree=3)
        relu = ActivationFactory.create('relu', degree=3)
        tanh = ActivationFactory.create('tanh', degree=3)
        
        self.assertIsNotNone(sigmoid)
        self.assertIsNotNone(relu)
        self.assertIsNotNone(tanh)


class TestLogisticRegressionModel(unittest.TestCase):
    """Test logistic regression model."""
    
    def setUp(self):
        """Set up test model."""
        self.model = LogisticRegression(input_dim=10)
        self.weights = np.array([0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.2, -0.4, 0.1, 0.35],
                              dtype=np.float32)
        self.model.load_weights(self.weights, bias=0.1)
    
    def test_model_loading(self):
        """Test model weight loading."""
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        np.testing.assert_array_equal(self.model.weights, self.weights)
        self.assertEqual(self.model.bias, 0.1)
    
    def test_plaintext_prediction(self):
        """Test plaintext prediction."""
        test_input = np.random.randn(10).astype(np.float32)
        prediction = self.model.predict_plaintext(test_input)[0]
        
        # Prediction should be between 0 and 1 (sigmoid output)
        self.assertGreater(prediction, 0)
        self.assertLess(prediction, 1)
    
    def test_encrypted_prediction(self):
        """Test encrypted prediction."""
        context = FHEContext(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[60, 40, 40, 60],
            scale_bits=40,
        )
        
        test_input = np.random.randn(10).astype(np.float32) * 0.5
        
        # Plaintext
        plaintext_pred = self.model.predict_plaintext(test_input)[0]
        
        # Encrypted
        encrypted_input = context.encrypt(test_input)
        encrypted_pred = self.model.predict_encrypted(encrypted_input)
        encrypted_pred_decrypted = context.decrypt(encrypted_pred)
        
        # Should be close (within FHE noise bounds)
        self.assertAlmostEqual(plaintext_pred, encrypted_pred_decrypted, places=1)


class TestNeuralNetwork(unittest.TestCase):
    """Test simple neural network model."""
    
    def setUp(self):
        """Set up test network."""
        self.network = SimpleNeuralNetwork(input_dim=10)
        self.network.add_layer(output_dim=8, activation='relu')
        self.network.add_layer(output_dim=4, activation='sigmoid')
    
    def test_network_creation(self):
        """Test network construction."""
        self.assertEqual(len(self.network.layers), 2)
        self.assertEqual(self.network.layers[0].output_dim, 8)
        self.assertEqual(self.network.layers[1].output_dim, 4)
    
    def test_architecture_string(self):
        """Test architecture description."""
        arch = self.network.get_architecture_string()
        self.assertIn('Input: 10', arch)
        self.assertIn('Dense(8)', arch)
        self.assertIn('Dense(4)', arch)


if __name__ == '__main__':
    unittest.main(verbosity=2)
