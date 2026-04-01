"""
FHE Client - Encryption, Decryption, and Server Communication

Client-side operations:
1. Key generation (secret key stays local)
2. Encrypt input data
3. Send to server
4. Receive encrypted result
5. Decrypt result (only client can do this)

All secret keys remain on client - never transmitted.
"""

import logging
import base64
import numpy as np
import requests
import tenseal as ts
from typing import Optional, Tuple
import json

from src.fhe.context import FHEContext
from src.ml.models import LogisticRegression, SimpleNeuralNetwork

logger = logging.getLogger(__name__)


class FHEClient:
    """
    Client for encrypted ML inference.
    
    Responsibilities:
    - Key generation and management
    - Encryption/decryption
    - Communication with server
    - Result processing
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 30):
        """
        Initialize FHE client.
        
        Args:
            server_url: FHE server endpoint
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
        # Cryptographic context
        self.fhe_context: Optional[FHEContext] = None
        
        # Model
        self.model: Optional[LogisticRegression] = None
        
        logger.info(f"FHE Client initialized: server={server_url}")
    
    def setup_fhe(
        self,
        poly_modulus_degree: int = 8192,
        coeff_modulus_bits: list = None,
        scale_bits: int = 40,
    ):
        """
        Initialize FHE context with specified parameters.
        
        Args:
            poly_modulus_degree: Polynomial modulus (2048-32768, power of 2)
            coeff_modulus_bits: Modulus chain coefficients
            scale_bits: Encoding scale bits
        """
        if coeff_modulus_bits is None:
            coeff_modulus_bits = [60, 40, 40, 60]
        
        self.fhe_context = FHEContext(
            poly_modulus_degree=poly_modulus_degree,
            coeff_modulus_bits=coeff_modulus_bits,
            scale_bits=scale_bits,
        )
        
        logger.info(
            f"FHE setup complete: "
            f"degree={poly_modulus_degree}, "
            f"scale_bits={scale_bits}"
        )
    
    def exchange_keys_with_server(self):
        """
        Send public context to server for initialization.
        
        WARNING: Only public context is sent (no secret key).
        """
        if self.fhe_context is None:
            raise ValueError("FHE context not initialized. Call setup_fhe() first.")
        
        # Serialize public context
        public_context_bytes = self.fhe_context.serialize_public_context()
        public_context_b64 = base64.b64encode(public_context_bytes).decode('utf-8')
        
        # Send to server
        try:
            response = requests.post(
                f"{self.server_url}/initialize",
                json={
                    'fhe_context_bytes': public_context_b64,
                    'model_version': 'v1',
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            logger.info(f"Key exchange completed with server: {response.json()}")
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"Key exchange failed: {str(e)}")
            raise
    
    def load_model(self, model_path: str = None):
        """
        Load a logistic regression model (optionally from file).
        
        Args:
            model_path: Path to model weights (can be numpy or JSON)
        """
        # For now, create a model, later can load from file
        self.model = LogisticRegression(input_dim=10)
        
        if model_path:
            # Load from file
            import os
            if model_path.endswith('.npy'):
                weights = np.load(model_path)
                self.model.load_weights(weights, bias=0.0)
            elif model_path.endswith('.json'):
                with open(model_path, 'r') as f:
                    data = json.load(f)
                    self.model.load_weights(data['weights'], data['bias'])
        else:
            # Default weights (for testing)
            weights = np.random.randn(10) * 0.1
            self.model.load_weights(weights, bias=0.0)
            logger.warning("Using random model weights (no model_path provided)")
        
        logger.info("Model loaded on client")
    
    def encrypt_data(self, data: np.ndarray) -> bytes:
        """
        Encrypt data locally.
        
        Args:
            data: Input array (shape: input_dim or batch)
            
        Returns:
            Serialized ciphertext bytes
        """
        if self.fhe_context is None:
            raise ValueError("FHE context not initialized")
        
        data = np.array(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1)
        
        encrypted = self.fhe_context.encrypt(data)
        ciphertext_bytes = self.fhe_context.serialize_ciphertext(encrypted)
        
        logger.debug(f"Encrypted data: input_shape={data.shape}, ciphertext_size={len(ciphertext_bytes)} bytes")
        
        return ciphertext_bytes
    
    def decrypt_result(self, ciphertext_bytes: bytes) -> np.ndarray:
        """
        Decrypt result from server.
        
        Args:
            ciphertext_bytes: Serialized encrypted result from server
            
        Returns:
            Decrypted plaintext numpy array
        """
        if self.fhe_context is None:
            raise ValueError("FHE context not initialized")
        
        encrypted_result = FHEContext.deserialize_ciphertext(ciphertext_bytes, self.fhe_context)
        plaintext = self.fhe_context.decrypt(encrypted_result)
        
        logger.debug(f"Decrypted result: shape={plaintext.shape}")
        
        return plaintext
    
    def infer(self, data: np.ndarray) -> np.ndarray:
        """
        Full encrypted inference pipeline:
        1. Encrypt data locally
        2. Send to server
        3. Receive encrypted result
        4. Decrypt result
        
        Args:
            data: Input features (shape: input_dim)
            
        Returns:
            Decrypted prediction
        """
        if self.fhe_context is None:
            raise ValueError("FHE context not initialized. Call setup_fhe() first.")
        
        # Step 1: Encrypt locally
        logger.info(f"Starting encrypted inference on input shape {data.shape}")
        ciphertext_bytes = self.encrypt_data(data)
        ciphertext_b64 = base64.b64encode(ciphertext_bytes).decode('utf-8')
        
        # Step 2: Send to server
        logger.info("Sending encrypted data to server...")
        try:
            response = requests.post(
                f"{self.server_url}/infer",
                json={
                    'ciphertext': ciphertext_b64,
                    'model_version': 'v1',
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Server inference request failed: {str(e)}")
            raise
        
        result_data = response.json()
        logger.info(f"Received encrypted result from server: {result_data['status']}")
        
        # Step 3: Decrypt locally
        result_b64 = result_data['result_ciphertext']
        result_ciphertext_bytes = base64.b64decode(result_b64)
        plaintext = self.decrypt_result(result_ciphertext_bytes)
        
        logger.info(f"Decrypted prediction: {plaintext}")
        
        return plaintext
    
    def get_server_status(self) -> dict:
        """Get server health status."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get server status: {str(e)}")
            return {'status': 'offline', 'error': str(e)}
    
    def get_server_config(self) -> dict:
        """Get server FHE configuration."""
        try:
            response = requests.get(f"{self.server_url}/config", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get server config: {str(e)}")
            return {'error': str(e)}


class ClientInferenceSession:
    """
    High-level session for end-to-end encrypted inference.
    
    Manages the entire lifecycle:
    1. Setup FHE
    2. Load model
    3. Key exchange
    4. Inference
    """
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """Initialize session."""
        self.client = FHEClient(server_url=server_url)
        self.is_initialized = False
    
    def initialize(self, skip_server_check: bool = False):
        """Initialize for inference."""
        logger.info("Initializing client session...")
        
        # Setup FHE
        self.client.setup_fhe()
        
        # Load model
        self.client.load_model()
        
        # Key exchange with server
        if not skip_server_check:
            try:
                status = self.client.get_server_status()
                if status.get('status') != 'healthy':
                    logger.warning(f"Server may not be ready: {status}")
            except Exception as e:
                logger.warning(f"Could not check server status: {e}")
        
        self.client.exchange_keys_with_server()
        
        self.is_initialized = True
        logger.info("Client session initialized and ready for inference")
    
    def run_inference(self, data: np.ndarray) -> np.ndarray:
        """Run encrypted inference."""
        if not self.is_initialized:
            raise ValueError("Session not initialized. Call initialize() first.")
        
        return self.client.infer(data)
