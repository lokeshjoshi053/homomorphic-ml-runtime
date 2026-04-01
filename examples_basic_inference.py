"""
End-to-End FHE ML Inference Example

Demonstrates the full pipeline:
1. Setup FHE context and load model locally
2. Create sample data and encrypt it
3. Perform encrypted inference
4. Decrypt result

Can run in standalone mode (client + server in same process for testing).
"""

import sys
import numpy as np
import logging
from typing import Tuple

# Setup path
sys.path.insert(0, '/Users/lokeshjoshi/Documents/Code Bases/phe-interface')

from src.fhe.context import FHEContext
from src.ml.models import LogisticRegression
from src.server.inference import EncryptedInferenceServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_standalone_inference():
    """
    Standalone example: Client + Server in same process.
    
    This demonstrates the FHE pipeline without network communication.
    """
    logger.info("=" * 70)
    logger.info("FHE ML INFERENCE EXAMPLE (Standalone)")
    logger.info("=" * 70)
    
    # ====== CLIENT SETUP ======
    logger.info("\n[CLIENT] Setting up FHE context...")
    
    client_context = FHEContext(
        poly_modulus_degree=8192,
        coeff_modulus_bits=[60, 40, 40, 60],
        scale_bits=40,
    )
    
    print(f"FHE Parameters:\n{client_context.get_parameters_summary()}\n")
    
    # ====== MODEL SETUP ======
    logger.info("[CLIENT] Loading logistic regression model...")
    
    # Create and load model
    model = LogisticRegression(input_dim=10)
    
    # Generate synthetic model weights (in practice, load from file)
    weights = np.array([
        0.5, -0.3, 0.2, 0.4, -0.1,
        0.3, 0.2, -0.4, 0.1, 0.35
    ], dtype=np.float32)
    bias = 0.1
    
    model.load_weights(weights, bias)
    logger.info(f"Model loaded: weights shape={weights.shape}, bias={bias}")
    
    # ====== GENERATE TEST DATA ======
    logger.info("\n[CLIENT] Generating test data...")
    
    # Create synthetic test samples
    num_samples = 3
    test_data = []
    
    for i in range(num_samples):
        # Generate random features (normalized to ~[-1, 1])
        sample = np.random.randn(10) * 0.5
        test_data.append(sample)
    
    test_data = np.array(test_data, dtype=np.float32)
    logger.info(f"Generated {num_samples} test samples, shape={(num_samples, 10)}")
    
    # ====== PLAINTEXT BASELINE ======
    logger.info("\n[PLAINTEXT] Computing baseline predictions (for comparison)...")
    
    plaintext_predictions = model.predict_plaintext(test_data)
    
    for i, pred in enumerate(plaintext_predictions):
        logger.info(f"  Sample {i+1}: {pred:.6f}")
    
    # ====== ENCRYPTED INFERENCE ======
    logger.info("\n[ENCRYPTED] Performing encrypted inference...")
    logger.info("-" * 70)
    
    encrypted_predictions = []
    
    for i, sample in enumerate(test_data):
        logger.info(f"\n  Processing sample {i+1}/{num_samples}...")
        
        # Step 1: Encrypt data
        logger.info(f"    1. Encrypting input...")
        encrypted_input = client_context.encrypt(sample)
        ciphertext_size = len(client_context.serialize_ciphertext(encrypted_input))
        logger.info(f"       Ciphertext size: {ciphertext_size} bytes")
        
        # Step 2: Remote inference (simulated on server)
        logger.info(f"    2. Performing encrypted inference...")
        encrypted_prediction = model.predict_encrypted(encrypted_input)
        
        # Step 3: Decrypt result
        logger.info(f"    3. Decrypting result...")
        decrypted_prediction = client_context.decrypt(encrypted_prediction)
        encrypted_predictions.append(decrypted_prediction)
        
        logger.info(f"       Predicted: {decrypted_prediction:.6f}")
        logger.info(f"       Plaintext: {plaintext_predictions[i]:.6f}")
        
        # Calculate error
        error = abs(decrypted_prediction - plaintext_predictions[i])
        logger.info(f"       Error: {error:.6e}")
    
    # ====== SUMMARY ======
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    logger.info("\nComparison of Plaintext vs Encrypted Predictions:")
    logger.info("\nSample | Plaintext | Encrypted | Error")
    logger.info("-------|-----------|-----------|-------")
    
    for i in range(num_samples):
        pt = plaintext_predictions[i]
        enc = encrypted_predictions[i]
        err = abs(pt - enc)
        logger.info(f"  {i+1}    | {pt:9.6f} | {enc:9.6f} | {err:.3e}")
    
    # Average error
    avg_error = np.mean([abs(plaintext_predictions[i] - encrypted_predictions[i]) 
                        for i in range(num_samples)])
    logger.info(f"\nAverage Error: {avg_error:.6e}")
    logger.info("(Error due to CKKS approximate arithmetic)\n")


def example_client_server_simulation():
    """
    Simulate client-server architecture (single process for illustration).
    
    Shows:
    1. Client encrypts
    2. Server performs inference (no decryption capability)
    3. Client decrypts result
    """
    logger.info("\n\n" + "=" * 70)
    logger.info("FHE ML INFERENCE - CLIENT/SERVER SIMULATION")
    logger.info("=" * 70)
    
    # ====== SERVER INITIALIZATION ======
    logger.info("\n[SERVER] Initializing encrypted inference server...")
    
    server = EncryptedInferenceServer(host="localhost", port=5000, debug=False)
    
    # Setup server FHE context (public only - no secret key)
    logger.info("[SERVER] Setting up FHE context (public)...")
    server_context = FHEContext(
        poly_modulus_degree=8192,
        coeff_modulus_bits=[60, 40, 40, 60],
        scale_bits=40,
    )
    server.set_fhe_context(server_context)
    
    # Load model on server
    logger.info("[SERVER] Loading model...")
    model = LogisticRegression(input_dim=10)
    weights = np.array([0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.2, -0.4, 0.1, 0.35],
                        dtype=np.float32)
    model.load_weights(weights, bias=0.1)
    server.set_model(model)
    
    logger.info("[SERVER] Ready to receive encrypted data\n")
    
    # ====== CLIENT OPERATION ======
    logger.info("[CLIENT] Setting up FHE context (full - with secret key)...")
    
    client_context = FHEContext(
        poly_modulus_degree=8192,
        coeff_modulus_bits=[60, 40, 40, 60],
        scale_bits=40,
    )
    
    # Generate test data
    test_sample = np.random.randn(10) * 0.5
    logger.info(f"[CLIENT] Generated test sample: shape={test_sample.shape}")
    
    # Encrypt
    logger.info("[CLIENT] Encrypting data...")
    encrypted_input = client_context.encrypt(test_sample)
    
    # Simulate server inference
    logger.info("[SERVER] Performing inference on encrypted data...")
    encrypted_result = server.model.predict_encrypted(encrypted_input)
    
    # Decrypt
    logger.info("[CLIENT] Decrypting result...")
    decrypted_result = client_context.decrypt(encrypted_result)
    
    # Verify
    plaintext_result = model.predict_plaintext(test_sample)
    error = abs(decrypted_result - plaintext_result)
    
    logger.info("\n" + "=" * 70)
    logger.info("Results:")
    logger.info(f"  Plaintext Prediction: {plaintext_result:.6f}")
    logger.info(f"  Encrypted Prediction: {decrypted_result:.6f}")
    logger.info(f"  Error: {error:.6e}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # Run standalone example
    example_standalone_inference()
    
    # Run client-server simulation
    example_client_server_simulation()
    
    logger.info("\n✓ All examples completed successfully!")
