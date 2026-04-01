"""
Polynomial Approximations for Activation Functions

Since FHE doesn't support non-polynomial operations directly,
we approximate activations using low-degree polynomials:
- Sigmoid ≈ polynomial of degree 3-7
- ReLU ≈ polynomial or Chebyshev approximation
- Tanh ≈ polynomial approximation

These dramatically reduce circuit depth and improve performance.
"""

import numpy as np
from typing import Callable, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ActivationApproximation:
    """Base class for activation function approximations."""
    
    def __init__(self, degree: int):
        """
        Args:
            degree: Polynomial degree (controls accuracy vs depth trade-off)
        """
        self.degree = degree
        self.coefficients = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply polynomial approximation to input."""
        raise NotImplementedError
    
    def get_coefficients(self) -> List[float]:
        """Get polynomial coefficients for homomorphic computation."""
        return self.coefficients.tolist()


class SigmoidApproximation(ActivationApproximation):
    """
    Sigmoid Approximation using Chebyshev Polynomials
    
    σ(x) = 1 / (1 + e^(-x))
    
    Approximates sigmoid in range [-4, 4] using Chebyshev expansion.
    Degree 3-5 provides good accuracy for ML inference.
    """
    
    def __init__(self, degree: int = 3, domain: Tuple[float, float] = (-4, 4)):
        """
        Args:
            degree: Polynomial degree (3 = good accuracy, 5+ = excellent BUT more multiplications)
            domain: Approximation domain (sigmoid saturates outside [-4, 4])
        """
        super().__init__(degree)
        self.domain = domain
        self.coefficients = self._compute_chebyshev_coefficients()
        
        logger.info(f"Sigmoid approximation: degree={degree}, domain={domain}")
    
    def _compute_chebyshev_coefficients(self) -> np.ndarray:
        """
        Compute Chebyshev approximation of sigmoid.
        
        Uses optimal Chebyshev polynomial for sigmoid in [-4, 4].
        Precomputed coefficients for common degrees.
        """
        # Precomputed Chebyshev coefficients for sigmoid in [-4, 4]
        precomputed = {
            3: np.array([0.5, 0.1975, 0.0, -0.018]),
            5: np.array([0.5, 0.2075, -0.0018, -0.018, 0.0, 0.00035]),
            7: np.array([0.5, 0.2159, -0.0048, -0.01678, 0.0, 0.000867, 0.0, -0.0000823])
        }
        
        if self.degree in precomputed:
            return precomputed[self.degree]
        else:
            # Fallback to degree-3 if not precomputed
            logger.warning(f"No precomputed coefficients for degree {self.degree}, using degree 3")
            return precomputed[3]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Chebyshev sigmoid approximation."""
        # Clip to domain
        x_clipped = np.clip(x, self.domain[0], self.domain[1])
        
        # Transform to [-1, 1] for Chebyshev
        x_normalized = 2 * (x_clipped - self.domain[0]) / (self.domain[1] - self.domain[0]) - 1
        
        # Evaluate polynomial
        result = np.polyval(self.coefficients, x_normalized)
        return result


class ReLUApproximation(ActivationApproximation):
    """
    ReLU Approximation using Polynomial
    
    ReLU(x) = max(0, x) ≈ polynomial in [-1, 2] range
    
    Polynomial ReLU approx: p(x) = 0.5 * x + 0.125 * x^3 (for degree 3)
    Provides reasonable approximation, especially for well-normalized inputs.
    """
    
    def __init__(self, degree: int = 3, domain: Tuple[float, float] = (-1, 2)):
        """
        Args:
            degree: Polynomial degree (3 = good for normalized inputs)
            domain: Approximation domain
        """
        super().__init__(degree)
        self.domain = domain
        self.coefficients = self._compute_coefficients()
        
        logger.info(f"ReLU approximation: degree={degree}, domain={domain}")
    
    def _compute_coefficients(self) -> np.ndarray:
        """Compute polynomial ReLU approximation."""
        # Cubic ReLU: smooth approximation with bounded derivative
        if self.degree >= 3:
            # p(x) = 0.5 * x + 0.125 * x^3 prevents negative saturation
            return np.array([0.125, 0.0, 0.5, 0.0])
        elif self.degree >= 1:
            # Linear: p(x) = 0.5 * x + 0.5 * ReLU compensation
            return np.array([0.5, 0.25])
        else:
            raise ValueError("Degree must be >= 1")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply polynomial ReLU approximation."""
        x_clipped = np.clip(x, self.domain[0], self.domain[1])
        result = np.polyval(self.coefficients, x_clipped)
        return result


class TanhApproximation(ActivationApproximation):
    """
    Tanh Approximation using Chebyshev Polynomials
    
    tanh(x) ≈ Chebyshev polynomial in [-2, 2] range
    Degree 3-5 provides good accuracy.
    """
    
    def __init__(self, degree: int = 3, domain: Tuple[float, float] = (-2, 2)):
        """
        Args:
            degree: Polynomial degree
            domain: Approximation domain
        """
        super().__init__(degree)
        self.domain = domain
        self.coefficients = self._compute_chebyshev_coefficients()
        
        logger.info(f"Tanh approximation: degree={degree}, domain={domain}")
    
    def _compute_chebyshev_coefficients(self) -> np.ndarray:
        """Compute Chebyshev approximation of tanh."""
        # Precomputed for common degrees
        precomputed = {
            3: np.array([0.9999, 0.0, -0.3331, 0.0]),
            5: np.array([0.99999, 0.0, -0.3346, 0.0, 0.0188, 0.0]),
        }
        
        if self.degree in precomputed:
            return precomputed[self.degree]
        else:
            logger.warning(f"No precomputed coefficients for degree {self.degree}, using degree 3")
            return precomputed[3]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Chebyshev tanh approximation."""
        x_clipped = np.clip(x, self.domain[0], self.domain[1])
        
        # Transform to [-1, 1]
        x_normalized = 2 * (x_clipped - self.domain[0]) / (self.domain[1] - self.domain[0]) - 1
        
        result = np.polyval(self.coefficients, x_normalized)
        return result


class ActivationFactory:
    """Factory for creating activation approximations."""
    
    _activations = {
        'sigmoid': SigmoidApproximation,
        'relu': ReLUApproximation,
        'tanh': TanhApproximation,
    }
    
    @staticmethod
    def create(activation_type: str, degree: int = 3) -> ActivationApproximation:
        """
        Create activation approximation.
        
        Args:
            activation_type: 'sigmoid', 'relu', or 'tanh'
            degree: Polynomial degree (3-7 typical)
            
        Returns:
            Activation approximation instance
        """
        if activation_type not in ActivationFactory._activations:
            raise ValueError(f"Unknown activation: {activation_type}")
        
        return ActivationFactory._activations[activation_type](degree=degree)


# Utility functions for polynomial evaluation on ciphertexts
def evaluate_polynomial_on_ciphertext(ciphertext, coefficients: List[float]):
    """
    Evaluate polynomial on encrypted ciphertext.
    
    Uses Horner's method for efficiency:
    p(x) = c0 + c1*x + c2*x^2 + ... 
         = c0 + x*(c1 + x*(c2 + ...))
    
    Args:
        ciphertext: Encrypted vector
        coefficients: Polynomial coefficients [c0, c1, c2, ...]
        
    Returns:
        Encrypted polynomial evaluation
    """
    if len(coefficients) == 0:
        raise ValueError("No coefficients provided")
    
    # Horner's method: works backwards from highest degree
    result = ciphertext * 0  # Zero ciphertext
    result = result + coefficients[-1]  # Start with highest coefficient
    
    for i in range(len(coefficients) - 2, -1, -1):
        result = result * ciphertext
        result = result + coefficients[i]
    
    return result


def polynomial_degree_from_multiplicative_depth(max_depth: int) -> int:
    """
    Determine maximum polynomial degree given multiplicative depth.
    
    Depth k allows polynomial of degree 2^k.
    Examples:
    - Depth 1: degree ≤ 2 (linear)
    - Depth 2: degree ≤ 4 (quartic)
    - Depth 3: degree ≤ 8
    - Depth 4: degree ≤ 16
    
    Args:
        max_depth: Maximum multiplicative depth available
        
    Returns:
        Maximum polynomial degree
    """
    return 2 ** max_depth
