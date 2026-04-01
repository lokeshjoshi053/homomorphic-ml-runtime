"""
Benchmarking suite for FHE ML inference.

Measures:
- Key generation time
- Encryption time
- Inference time (encrypted)
- Decryption time
- Ciphertext size
- Accuracy vs plaintext
"""

import sys
sys.path.insert(0, '/Users/lokeshjoshi/Documents/Code Bases/phe-interface')

import time
import numpy as np
import logging
from typing import Dict, List
import json

from src.fhe.context import FHEContext
from src.ml.models import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FHEBenchmark:
    """Comprehensive benchmarking for FHE system."""
    
    def __init__(self, num_trials: int = 5):
        """
        Args:
            num_trials: Number of repetitions for each benchmark
        """
        self.num_trials = num_trials
        self.results = {}
    
    def benchmark_keygen(self) -> Dict[str, float]:
        """Benchmark key generation."""
        logger.info("\n--- Benchmarking Key Generation ---")
        
        times = []
        for trial in range(self.num_trials):
            start = time.time()
            context = FHEContext(
                poly_modulus_degree=8192,
                coeff_modulus_bits=[60, 40, 40, 60],
                scale_bits=40,
            )
            end = time.time()
            times.append(end - start)
            logger.info(f"  Trial {trial+1}: {(end-start)*1000:.2f} ms")
        
        result = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
        
        logger.info(f"  Mean: {result['mean_ms']:.2f} ms (+/- {result['std_ms']:.2f} ms)")
        return result
    
    def benchmark_encryption(self, context: FHEContext, vector_size: int = 10) -> Dict[str, float]:
        """Benchmark encryption."""
        logger.info(f"\n--- Benchmarking Encryption (vector size={vector_size}) ---")
        
        test_vector = np.random.randn(vector_size).astype(np.float32)
        
        times = []
        for trial in range(self.num_trials):
            start = time.time()
            encrypted = context.encrypt(test_vector)
            end = time.time()
            times.append(end - start)
            logger.info(f"  Trial {trial+1}: {(end-start)*1000:.3f} ms")
        
        result = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
        
        logger.info(f"  Mean: {result['mean_ms']:.3f} ms")
        return result
    
    def benchmark_decryption(self, context: FHEContext, vector_size: int = 10) -> Dict[str, float]:
        """Benchmark decryption."""
        logger.info(f"\n--- Benchmarking Decryption (vector size={vector_size}) ---")
        
        test_vector = np.random.randn(vector_size).astype(np.float32)
        encrypted = context.encrypt(test_vector)
        
        times = []
        for trial in range(self.num_trials):
            start = time.time()
            decrypted = context.decrypt(encrypted)
            end = time.time()
            times.append(end - start)
            logger.info(f"  Trial {trial+1}: {(end-start)*1000:.3f} ms")
        
        result = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
        
        logger.info(f"  Mean: {result['mean_ms']:.3f} ms")
        return result
    
    def benchmark_inference(self, context: FHEContext, model: LogisticRegression) -> Dict[str, float]:
        """Benchmark encrypted inference."""
        logger.info("\n--- Benchmarking Encrypted Inference ---")
        
        test_vector = np.random.randn(model.input_dim).astype(np.float32)
        encrypted_input = context.encrypt(test_vector)
        
        times = []
        for trial in range(self.num_trials):
            start = time.time()
            encrypted_output = model.predict_encrypted(encrypted_input)
            end = time.time()
            times.append(end - start)
            logger.info(f"  Trial {trial+1}: {(end-start)*1000:.2f} ms")
        
        result = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
        
        logger.info(f"  Mean: {result['mean_ms']:.2f} ms")
        return result
    
    def benchmark_ciphertext_size(self, context: FHEContext, vector_size: int = 10) -> Dict[str, int]:
        """Benchmark ciphertext size."""
        logger.info(f"\n--- Measuring Ciphertext Size (vector size={vector_size}) ---")
        
        test_vector = np.random.randn(vector_size).astype(np.float32)
        encrypted = context.encrypt(test_vector)
        
        ciphertext_bytes = context.serialize_ciphertext(encrypted)
        size_kb = len(ciphertext_bytes) / 1024.0
        
        logger.info(f"  Size: {size_kb:.2f} KB ({len(ciphertext_bytes):,} bytes)")
        
        return {
            'bytes': len(ciphertext_bytes),
            'kb': size_kb,
        }
    
    def benchmark_accuracy(self, context: FHEContext, model: LogisticRegression, num_samples: int = 100) -> Dict[str, float]:
        """Benchmark accuracy: compare encrypted vs plaintext predictions."""
        logger.info(f"\n--- Measuring Accuracy (num_samples={num_samples}) ---")
        
        errors = []
        for i in range(num_samples):
            test_vector = np.random.randn(model.input_dim).astype(np.float32)
            
            # Plaintext prediction
            plaintext_pred = model.predict_plaintext(test_vector)[0]
            
            # Encrypted prediction
            encrypted_input = context.encrypt(test_vector)
            encrypted_output = model.predict_encrypted(encrypted_input)
            encrypted_pred = context.decrypt(encrypted_output)
            
            error = abs(plaintext_pred - encrypted_pred)
            errors.append(error)
        
        result = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
        }
        
        logger.info(f"  Mean error: {result['mean_error']:.6e}")
        logger.info(f"  Max error: {result['max_error']:.6e}")
        
        return result
    
    def run_full_benchmark(self) -> Dict:
        """Run all benchmarks."""
        logger.info("=" * 70)
        logger.info("FHE ML INFERENCE BENCHMARKING")
        logger.info("=" * 70)
        
        results = {}
        
        # Key generation
        results['keygen'] = self.benchmark_keygen()
        
        # Create context and model for subsequent benchmarks
        context = FHEContext(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[60, 40, 40, 60],
            scale_bits=40,
        )
        
        model = LogisticRegression(input_dim=10)
        weights = np.array([0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.2, -0.4, 0.1, 0.35],
                           dtype=np.float32)
        model.load_weights(weights, bias=0.1)
        
        # Encryption/Decryption
        results['encryption'] = self.benchmark_encryption(context, vector_size=10)
        results['decryption'] = self.benchmark_decryption(context, vector_size=10)
        
        # Inference
        results['inference'] = self.benchmark_inference(context, model)
        
        # Ciphertext size
        results['ciphertext_size'] = self.benchmark_ciphertext_size(context, vector_size=10)
        
        # Accuracy
        results['accuracy'] = self.benchmark_accuracy(context, model, num_samples=100)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print benchmarking summary."""
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70)
        
        logger.info("\nKey Generation:")
        logger.info(f"  {results['keygen']['mean_ms']:.2f} ms (±{results['keygen']['std_ms']:.2f} ms)")
        
        logger.info("\nEncryption (10-dim vector):")
        logger.info(f"  {results['encryption']['mean_ms']:.3f} ms")
        
        logger.info("\nDecryption (10-dim vector):")
        logger.info(f"  {results['decryption']['mean_ms']:.3f} ms")
        
        logger.info("\nEncrypted Inference:")
        logger.info(f"  {results['inference']['mean_ms']:.2f} ms")
        
        logger.info("\nCiphertext Size (10-dim vector):")
        logger.info(f"  {results['ciphertext_size']['kb']:.2f} KB")
        
        logger.info("\nAccuracy vs Plaintext:")
        logger.info(f"  Mean error: {results['accuracy']['mean_error']:.6e}")
        logger.info(f"  Max error: {results['accuracy']['max_error']:.6e}")
        
        # End-to-end timing
        e2e_time = (results['encryption']['mean_ms'] + 
                   results['inference']['mean_ms'] + 
                   results['decryption']['mean_ms'])
        logger.info(f"\nEnd-to-End Inference Time:")
        logger.info(f"  {e2e_time:.2f} ms (encryption + server inference + decryption)")
        
        logger.info("\n" + "=" * 70 + "\n")


def run_benchmarks():
    """Run full benchmark suite."""
    benchmark = FHEBenchmark(num_trials=5)
    results = benchmark.run_full_benchmark()
    
    # Save results to JSON
    with open('/Users/lokeshjoshi/Documents/Code Bases/phe-interface/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmarks()
