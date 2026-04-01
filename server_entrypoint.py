"""
Server Entrypoint - Starts FHE Inference Server

Initializes the Flask server with FHE context and model.
Handles graceful shutdown and error recovery.
"""

import os
import sys
import logging
import logging.config
import yaml
import numpy as np
from pathlib import Path

# Add source to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fhe.context import FHEContext
from src.ml.models import LogisticRegression
from src.server.inference import EncryptedInferenceServer


def setup_logging():
    """Configure structured logging for production."""
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'detailed',
                'level': 'INFO'
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'server.log',
                'formatter': 'json',
                'level': 'DEBUG'
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    })

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load FHE configuration from YAML."""
    config_path = Path(__file__).parent / 'fhe_config.yaml'
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def initialize_server(config: dict):
    """Initialize the inference server with FHE and model."""
    
    logger.info("Initializing FHE Inference Server...")
    
    # Get FHE parameters from config
    fhe_config = config.get('fhe', {})
    poly_modulus_degree = fhe_config.get('poly_modulus_degree', 8192)
    coeff_modulus_bits = fhe_config.get('coeff_modulus_bits', [60, 40, 40, 60])
    scale_bits = fhe_config.get('scale_bits', 40)
    
    # Initialize server
    server_config = config.get('server', {})
    server = EncryptedInferenceServer(
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 5000),
        debug=server_config.get('debug', False),
    )
    
    # Setup FHE context
    logger.info("Setting up FHE context...")
    fhe_context = FHEContext(
        poly_modulus_degree=poly_modulus_degree,
        coeff_modulus_bits=coeff_modulus_bits,
        scale_bits=scale_bits,
    )
    server.set_fhe_context(fhe_context)
    
    # Load model
    logger.info("Loading ML model...")
    ml_config = config.get('ml', {})
    input_dim = ml_config.get('input_dim', 10)
    model_type = ml_config.get('model_type', 'logistic_regression')
    
    if model_type == 'neural_network':
        from src.ml.models import create_simple_network
        model = create_simple_network(input_dim=input_dim)
        
        # Load random weights for demo
        for layer in model.layers:
            weights = np.random.randn(layer.output_dim, layer.input_dim) * 0.1
            bias = np.random.randn(layer.output_dim) * 0.1
            layer.load_weights(weights, bias)
        
        logger.info(f"Loaded neural network with {len(model.layers)} layers")
        
    else:  # logistic_regression
        model = LogisticRegression(input_dim=input_dim)
        
        # Load weights (generate random for demo, but can load from file)
        weights = np.random.randn(input_dim) * 0.1
        bias = 0.0
        model.load_weights(weights, bias=bias)
        
        logger.info("Loaded logistic regression model")
    
    server.set_model(model, version="v1", description=f"{model_type} model for FHE inference")
    
    logger.info("Server initialization complete")
    return server


def main():
    """Main entry point."""
    # Setup structured logging
    setup_logging()
    
    logger.info("=" * 70)
    logger.info("FHE ML INFERENCE SERVER")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize server
        server = initialize_server(config)
        
        # Start server
        logger.info("Starting server...")
        server.run()
    
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
