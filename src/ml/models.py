"""
ML Model implementations compatible with FHE.

Models must be converted to:
1. Linear layers (matrix multiply + add)
2. Polynomial activation approximations

Currently implements:
- Logistic Regression (linear model with sigmoid)
- Simple Dense Network (with polynomial activations)
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
import tenseal as ts
from src.ml.activations import ActivationFactory, evaluate_polynomial_on_ciphertext

logger = logging.getLogger(__name__)


class LogisticRegression:
    """
    Logistic Regression for FHE Inference
    
    Model: output = sigmoid(w^T * x + b)
    
    Converted to:
    1. Encrypted matrix multiplication: E(w^T) * E(x)
    2. Add bias: + b (plaintext for efficiency)
    3. Sigmoid approximation: polynomial(result)
    """
    
    def __init__(self, input_dim: int, activation_degree: int = 3):
        """
        Args:
            input_dim: Input feature dimension
            activation_degree: Degree of sigmoid polynomial approximation
        """
        self.input_dim = input_dim
        self.weights = None  # Shape: (1, input_dim) or (input_dim,)
        self.bias = None      # Shape: scalar
        self.activation = ActivationFactory.create('sigmoid', degree=activation_degree)
        
        logger.info(f"LogisticRegression initialized with input_dim={input_dim}")
    
    def load_weights(self, weights: np.ndarray, bias: float):
        """
        Load pretrained weights.
        
        Args:
            weights: Shape (input_dim,) or (1, input_dim)
            bias: Scalar bias term
        """
        weights = np.array(weights, dtype=np.float32)
        
        if weights.ndim == 2:
            weights = weights.squeeze()
        
        if len(weights) != self.input_dim:
            raise ValueError(f"Weight dimension mismatch: {len(weights)} vs {self.input_dim}")
        
        self.weights = weights
        self.bias = float(bias)
        
        logger.info(f"Loaded weights: shape={self.weights.shape}, bias={self.bias}")
    
    def train_from_sklearn(self, model):
        """Load weights from scikit-learn LogisticRegression."""
        weights = model.coef_[0]
        bias = model.intercept_[0]
        self.load_weights(weights, bias)
    
    def predict_plaintext(self, x: np.ndarray) -> np.ndarray:
        """
        Standard plaintext prediction for reference/testing.
        
        Args:
            x: Input of shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            Predicted probabilities
        """
        if self.weights is None:
            raise ValueError("Model weights not loaded")
        
        x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Linear: w^T * x + b
        logits = x @ self.weights + self.bias
        
        # Sigmoid
        predictions = 1 / (1 + np.exp(-logits))
        
        return predictions
    
    def predict_encrypted(self, encrypted_input: ts.CKKSVector) -> ts.CKKSVector:
        """
        Encrypted prediction using FHE.
        
        Args:
            encrypted_input: Encrypted feature vector E(x)
            
        Returns:
            Encrypted prediction E(output)
        """
        if self.weights is None:
            raise ValueError("Model weights not loaded")
        
        # Linear transformation: w^T * x (encrypted * plaintext)
        logits = encrypted_input * self.weights
        
        # Add bias (plaintext)
        logits = logits + self.bias
        
        # Sigmoid approximation on encrypted data
        activation_coeffs = self.activation.get_coefficients()
        prediction = evaluate_polynomial_on_ciphertext(logits, activation_coeffs)
        
        return prediction


class DenseLayer:
    """Single dense layer for neural network."""
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = None):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: 'sigmoid', 'relu', 'tanh', or None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_type = activation
        self.activation = ActivationFactory.create(activation) if activation else None
        
        # Weights: (output_dim, input_dim)
        # Bias: (output_dim,)
        self.weights = None
        self.bias = None
    
    def load_weights(self, weights: np.ndarray, bias: np.ndarray):
        """Load layer weights and bias."""
        weights = np.array(weights, dtype=np.float32)
        bias = np.array(bias, dtype=np.float32)
        
        if weights.shape != (self.output_dim, self.input_dim):
            raise ValueError(f"Weight shape mismatch: {weights.shape} vs ({self.output_dim}, {self.input_dim})")
        
        if len(bias) != self.output_dim:
            raise ValueError(f"Bias dimension mismatch: {len(bias)} vs {self.output_dim}")
        
        self.weights = weights
        self.bias = bias
        
        logger.info(f"DenseLayer {self.input_dim}->{self.output_dim} weights loaded")
    
    def forward_plaintext(self, x: np.ndarray) -> np.ndarray:
        """Plaintext forward pass."""
        if self.weights is None:
            raise ValueError("Weights not loaded")
        
        x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Linear: W @ x + b
        output = self.weights @ x + self.bias.reshape(-1, 1)
        
        # Activation
        if self.activation:
            output = self.activation.forward(output)
        
        return output.squeeze()
    
    def forward_encrypted(self, encrypted_input) -> ts.CKKSVector:
        """Encrypted forward pass."""
        if self.weights is None:
            raise ValueError("Weights not loaded")
        
        # Linear: W @ x + b
        # For simplicity with TenSEAL, handling single vector case
        output = encrypted_input * self.weights[0]  # First output neuron
        
        for i in range(1, self.output_dim):
            output = output + (encrypted_input * self.weights[i])
        
        output = output + self.bias
        
        # Activation
        if self.activation:
            activation_coeffs = self.activation.get_coefficients()
            output = evaluate_polynomial_on_ciphertext(output, activation_coeffs)
        
        return output


class SimpleNeuralNetwork:
    """
    Simple feedforward network for encrypted inference.
    
    Example: [10] -> Dense(16, relu) -> Dense(8, relu) -> Dense(1, sigmoid)
    """
    
    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Input feature dimension
        """
        self.input_dim = input_dim
        self.layers: List[DenseLayer] = []
        self.multiplicative_depth = 0
        
        logger.info(f"SimpleNeuralNetwork initialized with input_dim={input_dim}")
    
    def add_layer(self, output_dim: int, activation: str = None) -> 'SimpleNeuralNetwork':
        """
        Add a dense layer to the network.
        
        Args:
            output_dim: Output dimension
            activation: 'sigmoid', 'relu', 'tanh', or None
            
        Returns:
            self (for chaining)
        """
        if len(self.layers) == 0:
            input_dim = self.input_dim
        else:
            input_dim = self.layers[-1].output_dim
        
        layer = DenseLayer(input_dim, output_dim, activation)
        self.layers.append(layer)
        
        # Depth: 1 per multiplication (matrix) + 1 per activation polynomial
        if activation:
            # Activation polynomial adds multiplicative depth
            self.multiplicative_depth += 1  # Simplified; depends on activation degree
        
        logger.info(f"Added layer: {input_dim} -> {output_dim} (activation={activation})")
        return self
    
    def load_weights_from_sklearn(self, sklearn_model):
        """Load weights from sklearn neural network (MLPClassifier)."""
        if len(sklearn_model.coefs_) != len(self.layers):
            raise ValueError("Layer count mismatch with sklearn model")
        
        for i, layer in enumerate(self.layers):
            layer.load_weights(sklearn_model.coefs_[i].T, sklearn_model.intercepts_[i])
        
        logger.info("Loaded weights from sklearn model")
    
    def forward_plaintext(self, x: np.ndarray) -> np.ndarray:
        """Plaintext forward pass."""
        output = np.array(x, dtype=np.float32)
        
        for layer in self.layers:
            output = layer.forward_plaintext(output)
        
        return output
    
    def forward_encrypted(self, encrypted_input: ts.CKKSVector) -> ts.CKKSVector:
        """Encrypted forward pass."""
        output = encrypted_input
        
        for i, layer in enumerate(self.layers):
            output = layer.forward_encrypted(output)
            logger.debug(f"Completed encrypted layer {i+1}/{len(self.layers)}")
        
        return output
    
    def get_architecture_string(self) -> str:
        """Get readable network architecture."""
        arch = f"Input: {self.input_dim}\n"
        for i, layer in enumerate(self.layers):
            arch += f"  Layer {i+1}: Dense({layer.output_dim}), activation={layer.activation_type}\n"
        arch += f"Max multiplicative depth: {self.multiplicative_depth}"
        return arch


def create_logistic_regression_model(input_dim: int = 10) -> LogisticRegression:
    """Factory function for logistic regression."""
    return LogisticRegression(input_dim)


def create_simple_network(input_dim: int = 10, hidden_dims: List[int] = None) -> SimpleNeuralNetwork:
    """
    Factory function for simple neural network.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions with activations
                    Example: [(16, 'relu'), (8, 'relu'), (1, 'sigmoid')]
                    
    Returns:
        Configured neural network
    """
    if hidden_dims is None:
        hidden_dims = [(16, 'relu'), (8, 'relu'), (1, 'sigmoid')]
    
    network = SimpleNeuralNetwork(input_dim)
    for output_dim, activation in hidden_dims:
        network.add_layer(output_dim, activation)
    
    return network
