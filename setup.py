"""
Setup script for FHE ML Inference System
"""

from setuptools import setup, find_packages

setup(
    name="fhe-ml-inference",
    version="1.0.0",
    description="Privacy-preserving machine learning inference using Fully Homomorphic Encryption",
    author="Privacy Research Team",
    author_email="contact@example.com",
    license="MIT",
    
    packages=find_packages(),
    
    python_requires=">=3.11",
    
    install_requires=[
        "tenseal>=0.3.14",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "benchmark": [
            "matplotlib>=3.7.0",
            "tqdm>=4.66.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "fhe-server=server_entrypoint:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    
    keywords="FHE homomorphic encryption machine learning privacy",
    
    project_urls={
        "Documentation": "https://github.com/...",
        "Source": "https://github.com/...",
        "Issues": "https://github.com/.../issues",
    },
)
