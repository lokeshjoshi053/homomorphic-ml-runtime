#!/bin/bash
# Git setup and push script for homomorphic-ml-runtime

cd "/Users/lokeshjoshi/Documents/Code Bases/phe-interface"

echo "=== Checking Git Configuration ==="
echo "Current global git config:"
git config --global --list | grep -E "(user.name|user.email)" || echo "No global config found"

echo ""
echo "Setting up git user (if not set):"
git config --global user.name "lokeshjoshi053" || echo "Failed to set user.name"
git config --global user.email "lokeshjoshi053@users.noreply.github.com" || echo "Failed to set user.email"

echo ""
echo "Verifying git config:"
git config --global user.name
git config --global user.email

echo ""
echo "=== Initializing Git Repository ==="
if [ ! -d ".git" ]; then
    echo "Initializing new git repository..."
    git init
else
    echo "Git repository already exists"
fi

echo ""
echo "=== Setting up .gitignore ==="
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
.env.*.local
*.log

# OS
.DS_Store
Thumbs.db

# Project specific
models/
data/
*.pickle
server.log
test_context.py
EOF

echo ""
echo "=== Adding Remote Origin ==="
git remote add origin git@github.com:lokeshjoshi053/homomorphic-ml-runtime.git 2>/dev/null || \
git remote set-url origin git@github.com:lokeshjoshi053/homomorphic-ml-runtime.git

echo "Remote origin set to:"
git remote -v

echo ""
echo "=== Checking Repository Status ==="
git status

echo ""
echo "=== Staging Files ==="
git add .

echo ""
echo "=== Creating Commits ==="

# Check if there are any commits already
if git log --oneline -1 >/dev/null 2>&1; then
    echo "Repository has existing commits. Adding new changes..."
    git commit -m "feat: Complete FHE ML inference system with production features

- Production-grade encrypted ML inference server
- Neural network support with polynomial activations
- Model registry and versioning system
- Advanced monitoring, metrics, and health checks
- Comprehensive client-server architecture
- Request tracing and structured logging
- Docker containerization support
- Extensive testing and benchmarking suite"
else
    echo "Creating initial commit..."
    git commit -m "feat: Initial commit - Homomorphic ML Runtime

Complete implementation of privacy-preserving machine learning:
- FHE-based encrypted inference using CKKS scheme
- Client-server architecture for secure computation
- Support for neural networks and logistic regression
- Production-ready features: monitoring, health checks, metrics
- Comprehensive testing and documentation"
fi

echo ""
echo "=== Pushing to GitHub ==="
echo "Pushing to main branch..."
git branch -M main
git push -u origin main

echo ""
echo "=== Success! ==="
echo "Repository pushed to: https://github.com/lokeshjoshi053/homomorphic-ml-runtime"
echo "Git status:"
git status
echo "Recent commits:"
git log --oneline -5