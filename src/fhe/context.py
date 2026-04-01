"""
Core FHE Context and CKKS Scheme Implementation

This module provides the fundamental CKKS scheme operations:
- Key generation (public & secret keys)
- Encryption/Decryption
- Add, multiply, rescale operations
- Noise management and parameter tuning
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import tenseal as ts

logger = logging.getLogger(__name__)


class FHEContext:
    """
    CKKS Scheme Context Manager
    
    Manages the CKKS scheme setup, key generation, and all cryptographic operations.
    All encryption/decryption must happen within this context.
    
    SECURITY: Secret keys NEVER leave the client.
    """
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_modulus_bits: List[int] = None,
        scale_bits: int = 40,
        security_level: int = 128
    ):
        """
        Initialize CKKS context.
        
        Args:
            poly_modulus_degree: Polynomial modulus (must be power of 2)
                                Options: 2048, 4096, 8192, 16384, 32768
                                Higher = more security & capacity, but slower
            coeff_modulus_bits: Bit sizes for moduli chain
                               Affects both security and depth of computation
            scale_bits: Bit size for the encoding scale
                       Balances precision vs. noise
            security_level: Target security in bits (128 is standard)
        """
        
        if coeff_modulus_bits is None:
            # Default moduli for ~128-bit security with good depth
            coeff_modulus_bits = [60, 40, 40, 60]
        
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_modulus_bits = coeff_modulus_bits
        self.scale_bits = scale_bits
        self.security_level = security_level
        
        # Initialize TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_modulus_bits
        )
        
        # Configure scale
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** scale_bits
        
        # Track noise growth
        self.noise_budget = None
        self.max_multiplicative_depth = len(coeff_modulus_bits) - 1
        
        logger.info(
            f"FHE Context initialized: "
            f"degree={poly_modulus_degree}, "
            f"depth={self.max_multiplicative_depth}, "
            f"scale_bits={scale_bits}"
        )
    
    def get_keys(self) -> Tuple[bytes, bytes]:
        """
        Generate and export public/secret key pair.
        
        Returns:
            (public_key_bytes, secret_key_bytes)
            
        SECURITY CRITICAL:
        - Public key: Safe to share with server
        - Secret key: MUST remain on client, NEVER transmitted
        """
        public_key = self.context.public_key()
        secret_key = self.context.secret_key()
        
        public_key_bytes = public_key.serialize()
        secret_key_bytes = secret_key.serialize()
        
        logger.info("Generated CKKS key pair")
        return public_key_bytes, secret_key_bytes
    
    def serialize_public_context(self) -> bytes:
        """
        Export public context (no secret key) for server communication.
        
        Server uses this to work with ciphertexts without decryption capability.
        """
        return self.context.serialize(save_secret_key=False)
    
    def serialize_full_context(self) -> bytes:
        """
        Export full context with secret key. 
        
        WARNING: Only for client-side persistence. Never transmit.
        """
        return self.context.serialize(save_secret_key=True)
    
    @staticmethod
    def load_public_context(context_bytes: bytes) -> 'FHEContext':
        """Load a public context (no secret key) from bytes."""
        context = FHEContext.__new__(FHEContext)
        context.context = ts.context_from(context_bytes)
        logger.info("Loaded public FHE context")
        return context
    
    @staticmethod
    def load_full_context(context_bytes: bytes) -> 'FHEContext':
        """Load full context with secret key from bytes."""
        context = FHEContext.__new__(FHEContext)
        context.context = ts.context_from(context_bytes)
        logger.info("Loaded full FHE context with secret key")
        return context
    
    def encrypt(self, plaintext: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt plaintext vector using CKKS scheme.
        
        Args:
            plaintext: 1D numpy array of floating-point values
                      Values should be normalized to [-1, 1] for better precision
        
        Returns:
            Encrypted vector (ciphertext)
            
        E(f(x)) property: Operations on ciphertext preserve homomorphic property
        """
        if not isinstance(plaintext, np.ndarray):
            plaintext = np.array(plaintext)
        
        plaintext = plaintext.astype(np.float32)
        ciphertext = ts.ckks_vector(self.context, plaintext)
        
        logger.debug(f"Encrypted vector of size {len(plaintext)}")
        return ciphertext
    
    def decrypt(self, ciphertext: ts.CKKSVector) -> np.ndarray:
        """
        Decrypt ciphertext to plaintext.
        
        Can only be performed on client with secret key.
        Server cannot decrypt.
        
        Args:
            ciphertext: Encrypted vector
            
        Returns:
            Decrypted plaintext as numpy array
        """
        plaintext = ciphertext.decrypt()
        return np.array(plaintext, dtype=np.float32)
    
    def add(self, c1: ts.CKKSVector, c2: ts.CKKSVector) -> ts.CKKSVector:
        """
        Homomorphic addition: E(x + y) from E(x) and E(y)
        
        No noise growth (much faster than multiplication).
        """
        return c1 + c2
    
    def add_plaintext(self, ciphertext: ts.CKKSVector, plaintext: np.ndarray) -> ts.CKKSVector:
        """Add plaintext to ciphertext: E(x + p) from E(x) and p"""
        plaintext = np.array(plaintext, dtype=np.float32)
        return ciphertext + plaintext
    
    def multiply(self, c1: ts.CKKSVector, c2: ts.CKKSVector) -> ts.CKKSVector:
        """
        Homomorphic multiplication: E(x * y) from E(x) and E(y)
        
        WARNING: Increases noise and consumes multiplicative depth.
        Check noise budget before using in deep circuits.
        """
        result = c1 * c2
        # Automatic rescaling in TenSEAL
        return result
    
    def multiply_plaintext(self, ciphertext: ts.CKKSVector, plaintext: np.ndarray) -> ts.CKKSVector:
        """Multiply ciphertext by plaintext: E(x * p) from E(x) and p"""
        plaintext = np.array(plaintext, dtype=np.float32)
        return ciphertext * plaintext
    
    def rescale(self, ciphertext: ts.CKKSVector) -> ts.CKKSVector:
        """
        Manually rescale to reduce noise and free up moduli.
        
        TenSEAL handles this automatically after multiplication,
        but can be called explicitly for fine-grained control.
        """
        # TenSEAL handles rescaling automatically
        return ciphertext
    
    def get_noise_budget(self, ciphertext: ts.CKKSVector) -> Optional[float]:
        """
        Get remaining noise budget (in bits).
        
        Returns: Bits remaining before decryption fails
        None if secret key not available (can't compute on server)
        """
        # Note: TenSEAL doesn't directly expose noise budget
        # This would require SEAL C++ library
        # Returning None for now
        return None
    
    def serialize_ciphertext(self, ciphertext: ts.CKKSVector) -> bytes:
        """
        Serialize ciphertext for transmission (e.g., to server).
        
        Safe to transmit - contains no secret information.
        Size: typically 1-10 KB depending on vector size and complexity.
        """
        return ciphertext.serialize()
    
    @staticmethod
    def deserialize_ciphertext(ciphertext_bytes: bytes, context: 'FHEContext') -> ts.CKKSVector:
        """Deserialize ciphertext from bytes."""
        return ts.ckks_vector_from(context.context, ciphertext_bytes)
    
    def get_parameters_summary(self) -> dict:
        """Get summary of cryptographic parameters for logging/debugging."""
        return {
            'poly_modulus_degree': self.poly_modulus_degree,
            'coeff_modulus_bits': self.coeff_modulus_bits,
            'scale_bits': self.scale_bits,
            'security_level': self.security_level,
            'max_multiplicative_depth': self.max_multiplicative_depth,
        }
