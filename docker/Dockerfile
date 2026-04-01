# FHE Inference Server - Python/TenSEAL
# Production-grade Dockerfile for encrypted ML inference

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TenSEAL and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY fhe_config.yaml .
COPY server_entrypoint.py .

# Create non-root user for security
RUN useradd -m -u 1000 fheuser && chown -R fheuser:fheuser /app
USER fheuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)"

# Expose port
EXPOSE 5000

# Run server
CMD ["python", "server_entrypoint.py"]
