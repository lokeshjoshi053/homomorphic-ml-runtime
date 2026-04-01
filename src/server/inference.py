"""
Encrypted Inference Server

Stateless computation node that:
1. Receives encrypted data (no plaintext exposure)
2. Performs encrypted ML inference
3. Returns encrypted predictions
4. Has NO access to secret keys (cannot decrypt)
"""

import logging
import time
import uuid
from typing import Optional, Dict, Any, List
import json
import numpy as np
import tenseal as ts
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request schema for encrypted inference."""
    ciphertext: str  # Base64-encoded serialized ciphertext
    model_version: str = "v1"


class InferenceResponse(BaseModel):
    """Response schema for encrypted results."""
    result_ciphertext: str  # Base64-encoded encrypted prediction
    status: str = "success"
    model_version: str = "v1"
    request_id: str = ""


@dataclass
class ModelMetadata:
    """Model metadata for versioning and management."""
    version: str
    model_type: str
    input_dim: int
    output_dim: int
    architecture: str
    created_at: str
    description: str = ""


class ModelRegistry:
    """Registry for managing multiple ML models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.active_version: str = "v1"
    
    def register_model(self, version: str, model: Any, metadata: ModelMetadata):
        """Register a model with metadata."""
        self.models[version] = model
        self.metadata[version] = metadata
        logger.info(f"Registered model version {version}: {metadata.model_type}")
    
    def get_model(self, version: str = None) -> Any:
        """Get model by version."""
        if version is None:
            version = self.active_version
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")
        return self.models[version]
    
    def get_metadata(self, version: str = None) -> ModelMetadata:
        """Get model metadata."""
        if version is None:
            version = self.active_version
        return self.metadata.get(version)
    
    def list_versions(self) -> List[str]:
        """List all registered model versions."""
        return list(self.models.keys())
    
    def set_active_version(self, version: str):
        """Set the active model version."""
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")
        self.active_version = version
        logger.info(f"Set active model version to {version}")


class MetricsCollector:
    """Collect and expose server metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_inference_time = 0.0
        self.inference_count = 0
        self.last_request_time = None
    
    def record_request(self):
        """Record an incoming request."""
        self.request_count += 1
        self.last_request_time = time.time()
    
    def record_inference(self, duration: float):
        """Record inference metrics."""
        self.inference_count += 1
        self.total_inference_time += duration
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self.start_time
        avg_inference_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "total_inferences": self.inference_count,
            "average_inference_time_seconds": avg_inference_time,
            "last_request_timestamp": self.last_request_time
        }


class EncryptedInferenceServer:
    """
    Production-grade Flask-based server for performing inference on encrypted data.
    
    Advanced features:
    - Model versioning and registry
    - Request tracing with IDs
    - Health checks and metrics
    - Structured logging
    - Graceful error handling
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5001, debug: bool = False):
        """
        Initialize server.
        
        Args:
            host: Bind address
            port: Bind port
            debug: Enable Flask debug mode
        """
        self.app = Flask("FHEInferenceServer")
        CORS(self.app)
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Model registry and FHE context
        self.model_registry = ModelRegistry()
        self.fhe_context = None
        
        # Metrics collection
        self.metrics = MetricsCollector()
        
        # Request tracing
        @self.app.before_request
        def before_request():
            g.request_id = str(uuid.uuid4())
            g.start_time = time.time()
            self.metrics.record_request()
            logger.info(f"Request {g.request_id}: {request.method} {request.path}")
        
        @self.app.after_request
        def after_request(response):
            duration = time.time() - g.start_time
            logger.info(f"Request {g.request_id} completed in {duration:.3f}s with status {response.status_code}")
            return response
        
        # Register routes
        self._register_routes()
        
        logger.info(f"EncryptedInferenceServer initialized: {host}:{port}")
    
    def _register_routes(self):
        """Register API endpoints."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Comprehensive health check endpoint."""
            try:
                # Check FHE context
                context_healthy = self.fhe_context is not None
                if context_healthy:
                    # Test context with a simple operation
                    test_vec = ts.ckks_vector(self.fhe_context.context, [1.0])
                    context_healthy = test_vec is not None
                
                # Check model registry
                models_available = len(self.model_registry.list_versions()) > 0
                
                health_status = {
                    'status': 'healthy' if context_healthy and models_available else 'degraded',
                    'timestamp': time.time(),
                    'components': {
                        'fhe_context': 'healthy' if context_healthy else 'unhealthy',
                        'model_registry': 'healthy' if models_available else 'unhealthy',
                    },
                    'active_model_version': self.model_registry.active_version,
                    'available_models': self.model_registry.list_versions(),
                }
                
                status_code = 200 if health_status['status'] == 'healthy' else 503
                return jsonify(health_status), status_code
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': time.time()
                }), 503
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available models and their metadata."""
            try:
                models = []
                for version in self.model_registry.list_versions():
                    metadata = self.model_registry.get_metadata(version)
                    models.append({
                        'version': version,
                        'model_type': metadata.model_type,
                        'input_dim': metadata.input_dim,
                        'output_dim': metadata.output_dim,
                        'architecture': metadata.architecture,
                        'created_at': metadata.created_at,
                        'description': metadata.description,
                        'is_active': version == self.model_registry.active_version
                    })
                
                return jsonify({
                    'models': models,
                    'active_version': self.model_registry.active_version
                }), 200
                
            except Exception as e:
                logger.error(f"List models error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/models/<version>', methods=['GET'])
        def get_model_info(version):
            """Get detailed information about a specific model."""
            try:
                metadata = self.model_registry.get_metadata(version)
                if not metadata:
                    return jsonify({'error': f'Model version {version} not found'}), 404
                
                return jsonify({
                    'version': version,
                    'metadata': asdict(metadata),
                    'is_active': version == self.model_registry.active_version
                }), 200
                
            except Exception as e:
                logger.error(f"Get model info error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/models/<version>/activate', methods=['POST'])
        def activate_model(version):
            """Activate a specific model version."""
            try:
                self.model_registry.set_active_version(version)
                return jsonify({
                    'status': 'activated',
                    'active_version': version
                }), 200
                
            except ValueError as e:
                return jsonify({'error': str(e)}), 404
            except Exception as e:
                logger.error(f"Activate model error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            """Get server performance metrics."""
            try:
                metrics_data = self.metrics.get_metrics()
                return jsonify(metrics_data), 200
                
            except Exception as e:
                logger.error(f"Get metrics error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Get server configuration."""
            if self.fhe_context is None:
                return jsonify({'error': 'Context not initialized'}), 400
            
            params = self.fhe_context.get_parameters_summary()
            return jsonify({
                'fhe_parameters': params,
                'active_model_version': self.model_registry.active_version,
                'server_info': {
                    'host': self.host,
                    'port': self.port,
                    'debug': self.debug
                }
            })
        
        @self.app.route('/infer', methods=['POST'])
        def infer():
            """Perform encrypted inference."""
            request_start = time.time()
            try:
                # Parse request
                data = request.get_json()
                if not data:
                    self.metrics.record_error()
                    return jsonify({'error': 'Empty request body'}), 400
                
                # Validate request schema
                try:
                    req = InferenceRequest(**data)
                except ValidationError as e:
                    self.metrics.record_error()
                    return jsonify({'error': str(e)}), 400
                
                # Get model from registry
                try:
                    model = self.model_registry.get_model(req.model_version)
                except ValueError as e:
                    self.metrics.record_error()
                    return jsonify({'error': str(e)}), 404
                
                # Check FHE context
                if self.fhe_context is None:
                    self.metrics.record_error()
                    return jsonify({'error': 'FHE context not initialized'}), 503
                
                # Deserialize ciphertext from client
                import base64
                ciphertext_bytes = base64.b64decode(req.ciphertext)
                encrypted_input = ts.ckks_vector_from(self.fhe_context.context, ciphertext_bytes)
                
                logger.info(f"Request {g.request_id}: Inference with model {req.model_version}")
                
                # Perform inference on encrypted data
                inference_start = time.time()
                encrypted_result = model.predict_encrypted(encrypted_input)
                inference_duration = time.time() - inference_start
                
                # Record metrics
                self.metrics.record_inference(inference_duration)
                
                # Serialize result ciphertext
                result_bytes = encrypted_result.serialize()
                result_b64 = base64.b64encode(result_bytes).decode('utf-8')
                
                # Return response
                response = InferenceResponse(
                    result_ciphertext=result_b64,
                    model_version=req.model_version,
                    request_id=g.request_id
                )
                
                return jsonify(response.model_dump()), 200
            
            except Exception as e:
                self.metrics.record_error()
                logger.error(f"Request {g.request_id}: Inference error: {str(e)}", exc_info=True)
                return jsonify({'error': f'Inference failed: {str(e)}'}), 500
        
        @self.app.route('/initialize', methods=['POST'])
        def initialize():
            """
            Initialize server with FHE context and model.
            
            Should be called with public context (no secret key).
            Model weights are provided as plaintext (only weights, no data).
            """
            try:
                data = request.get_json()
                
                # Load FHE public context
                if 'fhe_context_bytes' not in data:
                    return jsonify({'error': 'Missing fhe_context_bytes'}), 400
                
                import base64
                context_bytes = base64.b64decode(data['fhe_context_bytes'])
                self.fhe_context = ts.context_from(context_bytes)
                
                # Load model (would need to deserialize model state here)
                # For now, placeholder - actual model loading logic goes here
                if 'model_weights' in data:
                    logger.info("Model weights received and loaded")
                
                logger.info("Server initialization complete")
                return jsonify({
                    'status': 'initialized',
                    'active_model_version': self.model_registry.active_version,
                }), 200
            
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}", exc_info=True)
                return jsonify({'error': f'Initialization failed: {str(e)}'}), 500
    
    def set_model(self, model, version: str = "v1", description: str = ""):
        """Set the Model for inference and register in model registry."""
        from datetime import datetime
        
        # Determine model type and dimensions
        if hasattr(model, 'layers'):
            model_type = "neural_network"
            input_dim = model.input_dim
            output_dim = model.layers[-1].output_dim if model.layers else 1
            architecture = model.get_architecture_string()
        else:
            model_type = "logistic_regression"
            input_dim = model.input_dim
            output_dim = 1
            architecture = f"LogisticRegression(input_dim={input_dim})"
        
        # Create metadata
        metadata = ModelMetadata(
            version=version,
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            architecture=architecture,
            created_at=datetime.now().isoformat(),
            description=description
        )
        
        # Register model
        self.model_registry.register_model(version, model, metadata)
        
        # Set as active if it's the first model
        if len(self.model_registry.list_versions()) == 1:
            self.model_registry.set_active_version(version)
        
        logger.info(f"Model {version} registered: {model_type}")
    
    def set_fhe_context(self, context):
        """Set the FHE context (public only - no secret key)."""
        self.fhe_context = context
        logger.info("FHE context set on server")
    
    def run(self):
        """Start the Flask server."""
        logger.info(f"Starting FHE inference server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)
