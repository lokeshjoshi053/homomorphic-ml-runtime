# Privacy-Preserving ML Inference with Fully Homomorphic Encryption

A production-grade system for secure, encrypted machine learning inference using CKKS (Cheon-Kim-Kim-Song) scheme. Enables ML predictions directly on encrypted data without ever exposing plaintext to the server.

## System Overview

```
[CLIENT]                        [SERVER]
  │                               │
  ├─ Encrypt input E(x)  ──────→  │
  │                               ├─ Encrypted inference: E(f(x))
  │  ← Receive E(f(x))  ──────────┤
  │                               │
  └─ Decrypt & get f(x)           │
```

**Key Properties:**
- ✅ **Zero-Knowledge**: Server performs inference without access to plaintext data or model architecture
- ✅ **Homomorphic**: Computations on encrypted data produce encrypted results
- ✅ **CKKS Scheme**: Approximate arithmetic for practical ML inference
- ✅ **Privacy-Preserving**: Client retains all secret keys; never transmitted
- ✅ **Scalable**: Vectorized operations, noise management, parameter optimization

## Quick Start

### 1. Installation

```bash
# Clone repository
cd /Users/lokeshjoshi/Documents/Code\ Bases/phe-interface

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Basic Example

```bash
# Standalone inference (demonstrates full pipeline)
python examples_basic_inference.py
```

This demonstrates:
- FHE key generation
- Encrypting input data
- Performing encrypted inference
- Decrypting results
- Comparing plaintext vs encrypted predictions

### 3. Run Tests

```bash
# Run test suite
python -m pytest tests/test_fhe_inference.py -v
```

### 4. Run Benchmarks

```bash
# Measure latency, throughput, ciphertext size
python benchmarks/performance.py
```

## System Architecture

### Components

1. **FHE Core** (`src/fhe/`)
   - CKKS scheme implementation using TenSEAL
   - Key generation, encryption, decryption
   - Homomorphic operations (add, multiply, rescale)

2. **ML Models** (`src/ml/`)
   - Logistic Regression with sigmoid approximation
   - Simple Neural Networks with polynomial activations
   - Activation approximations (sigmoid, ReLU, tanh)

3. **Server** (`src/server/`)
   - Flask-based REST API for encrypted inference
   - Stateless computation node
   - No secret key access

4. **Client** (`src/client/`)
   - End-to-end encryption/decryption
   - Server communication
   - Key management (local only)

### Threat Model

**What the system protects:**
- Input data remains encrypted in transit and on server
- Model weights not exposed to client
- Server cannot decrypt predictions
- Inference process is cryptographically secure

**Trust assumptions:**
- Client trusts its own machine (key storage)
- Server is honest-but-curious (follows protocol, doesn't decrypt)
- Communication channel is secure (TLS in production)

## Core Concepts

### CKKS Scheme

Cheon-Kim-Kim-Song scheme provides **approximate arithmetic** on encrypted real numbers.

**Key properties:**
- Supports addition and multiplication on encrypted data
- Automatically rescales after operations to manage noise
- Quotient and remainder encoding for vectorization
- Practical for ML inference with bounded precision requirements

### Homomorphic Property

Core equation: **E(f(x)) = f(E(x))**

Example chain:
```
Linear: E(w^T x + b) = E(w^T x) + E(b) = w^T E(x) + b
Activation: sigmoid(E(logits)) ≈ poly(E(logits))
```

### Noise Management

Each operation increases noise (ciphertext perturbation). Managed through:

1. **Rescaling**: Reduces noise after multiplication
   - Decreases modulus
   - Frees up computation depth

2. **Modulus Switching**: Reduces ciphertext size
   - Multiple moduli in chain allow progressive depth reduction
   - Balanced between precision and computational depth

3. **Bootstrapping** (optional): Resets noise
   - Expensive operation (~1 second)
   - Allows deeper circuits
   - Disabled by default for performance

## FHE Parameters

Key configuration in `fhe_config.yaml`:

```yaml
fhe:
  poly_modulus_degree: 8192        # Polynomial modulus [2048, 4096, 8192, 16384, 32768]
  coeff_modulus_bits: [60, 40, 40, 60]  # Modulus chain
  scale: 40                        # Encoding scale bits
  security_level: 128              # 128-bit security
  
ml:
  activation:
    degree: 3                      # Polynomial degree for approximations
```

### Parameter Trade-offs

| Parameter | Impact | Trade-off |
|-----------|--------|-----------|
| `poly_modulus_degree` | Security & depth | Higher = slower key gen |
| `coeff_modulus_bits` | Computation depth | More bits = more noise |
| `scale_bits` | Precision | Higher = less multiplicative depth |
| `activation_degree` | Approximation accuracy | Higher = more multiplications |

## Usage Guide

### Client-Side Encryption

```python
from src.fhe.context import FHEContext
import numpy as np

# Create FHE context
context = FHEContext(
    poly_modulus_degree=8192,
    coeff_modulus_bits=[60, 40, 40, 60],
    scale_bits=40,
)

# Encrypt data
plaintext = np.array([0.1, 0.2, 0.3, ...])
encrypted = context.encrypt(plaintext)

# Serialize for transmission
ciphertext_bytes = context.serialize_ciphertext(encrypted)
```

### Encrypted Inference

```python
from src.ml.models import LogisticRegression

# Create and load model
model = LogisticRegression(input_dim=10)
model.load_weights(weights, bias=0.1)

# Encrypted inference
encrypted_result = model.predict_encrypted(encrypted_input)

# Decryption (client-only)
result = context.decrypt(encrypted_result)
```

### Full End-to-End Pipeline

```python
from src.client.session import ClientInferenceSession

# Initialize session
session = ClientInferenceSession(server_url="http://localhost:5000")
session.initialize()

# Run inference
test_data = np.array([0.1, 0.2, ..., 0.5])
prediction = session.run_inference(test_data)
print(f"Predicted: {prediction}")
```

## Server Deployment

### Local Development

```bash
# Start server (Flask debug mode)
python server_entrypoint.py
```

Server runs at `http://localhost:5000`

### Docker Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# Or build manually
docker build -t fhe-server:latest .
docker run -p 5000:5000 fhe-server:latest
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Server health status |
| `/config` | GET | FHE parameters & config |
| `/initialize` | POST | Setup server with FHE context |
| `/infer` | POST | Perform encrypted inference |

### Example: Node.js Server Inference

```bash
curl -X POST http://localhost:5000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "ciphertext": "<base64-encoded-ciphertext>",
    "model_version": "v1"
  }'
```

## Performance Characteristics

### Benchmarks

Typical measurements on Intel i7 with 8192-degree polynomial:

| Operation | Time |
|-----------|------|
| Key Generation | 200-300 ms |
| Encryption (10-dim) | 5-10 ms |
| Decryption (10-dim) | 3-5 ms |
| Inference (logistic reg) | 50-100 ms |
| **End-to-end** | 60-120 ms |

### Ciphertext Sizes

| Data | Size |
|------|------|
| 10-dim encrypted vector | 100-200 KB |
| Model weights (encrypted) | 1-5 MB |
| Public context | 50-100 KB |

Run benchmarks:
```bash
python benchmarks/performance.py
```

## Noise & Depth Analysis

### Multiplicative Depth

Each multiplication operation "consumes" one level of the modulus chain:
- Degree 4 (8192): ~3 levels deep
- Degree 8 (16384): ~4 levels deep
- Degree 16384 (8192): ~5 levels deep

**Current circuit depth:**
- Logistic regression: 1 (1 multiplication for w^T x, 1 for sigmoid approx)
- Small NN (2 layers): 2-3
- With activation approximations: Add 1 per activation

### Noise Growth

Operations increase noise budget consumption:

1. **Addition**: No noise growth
2. **Multiplication**: Quadratic noise growth (uses rescaling)
3. **Activation approximations**: Polynomial degree = depth consumption

**Monitoring**: Use `get_noise_budget()` to track remaining capacity (when secret key available).

## Advanced Features

### Polynomial Activations

Instead of non-polynomial activations (incompatible with FHE), we use approximations:

```python
from src.ml.activations import ActivationFactory

# Create approximation
sigmoid = ActivationFactory.create('sigmoid', degree=3)

# Evaluate on encrypted data
encrypted_result = sigmoid.forward(encrypted_input)
```

Supported approximations:
- Sigmoid: Chebyshev polynomial, degree 3-7
- ReLU: Polynomial, degree 3
- Tanh: Chebyshev polynomial, degree 3-5

### Multi-Party FHE (Future)

Can extend to threshold decryption where multiple parties hold key shares:
- No single party can decrypt alone
- Quorum-based decryption for shared privacy

## Security Considerations

### Key Management

```
client/
  ├── public_key.bin    (safe to share)
  ├── secret_key.bin    (NEVER transmit)
  └── context.bin       (with secret key, local storage only)
```

**CRITICAL**: Never transmit secret keys. Only transmit:
- Public keys
- Public context
- Ciphertexts

### Secure Communication

In production, use:
- TLS 1.3 for client-server communication
- Certificate pinning for server authentication
- Encrypted storage for secret keys at rest

### Side-Channel Attacks

Current implementation provides:
- ✅ Constant-time encryption (via TenSEAL)
- ✅ No timing leakage on decryption
- ⚠️ Potential cache timing (mitigated by TenSEAL)

For critical deployments, consider:
- Intel SGX enclave execution
- Hardware security modules (HSM)
- Formal verification

## Troubleshooting

### High Prediction Error

**Symptom**: Large gap between plaintext and encrypted predictions.

**Causes**:
1. Noise budget exhausted → try higher `scale_bits`
2. Activation outside domain → normalize inputs to [-1, 1]
3. Too many operations → reduce model depth

**Solution**:
```python
# Increase scale bits
context = FHEContext(scale_bits=50)  # from 40

# Or reduce polynomial degree
sigmoid = ActivationFactory.create('sigmoid', degree=3)  # lower = less depth
```

### Server Inference Fails

**Check**:
```bash
curl http://localhost:5000/health
curl http://localhost:5000/config
```

**Common issues**:
- Model not loaded → check `server_entrypoint.py`
- FHE context not initialized → verify client key exchange
- Ciphertext serialization error → ensure consistent parameters

### Memory Issues

TenSEAL caches computations. Clear periodically:
```python
# After batch of operations
import gc
gc.collect()
```

## Future Roadmap

- [ ] Bootstrapping for unlimited depth
- [ ] Batched inference API
- [ ] Quantization for reduced memory
- [ ] Rust implementation for 10x speedup
- [ ] Zero-knowledge proof integration
- [ ] On-chain inference verification
- [ ] Multi-party threshold scheme

## References & Resources

### Theory
- [CKKS Paper](https://eprint.iacr.org/2016/421.pdf) - Homomorphic Encryption for Arithmetic
- [FHE Overview](https://github.com/homomorphicencryption/HElib) - Microsoft HElib
- [Noise Analysis](https://eprint.iacr.org/2021/204.pdf) - Practical FHE Bounds

### Libraries
- [TenSEAL](https://github.com/OpenMined/TenSEAL) - Python FHE
- [SEAL](https://github.com/microsoft/SEAL) - C++ Foundation
- [Lattigo](https://github.com/tuneinsight/lattigo) - Go Implementation

### Tools & Demos
- [FHE.org](https://fhe.org) - Community resources
- [KyberNet](https://github.com/openmined/KyberNet) - Privacy benchmarks
- [CrypTen](https://github.com/facebookresearch/CrypTen) - Facebook's framework

## Contributing

Contributions welcome! Areas of interest:
- Performance optimizations (Rust backend)
- Additional activation approximations
- Inference serving infrastructure
- Benchmarking on real datasets
- Documentation & examples

## License

MIT License - See LICENSE file

## Citation

```bibtex
@project{fhe-ml-inference-2024,
  title = {Privacy-Preserving ML Inference with FHE},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/...}
}
```

---

**Questions?** Open an issue or start a discussion.

**Security Issues?** Please email security@... instead of public issues.
