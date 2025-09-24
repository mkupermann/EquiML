#!/bin/bash
# EquiML Docker Build Script
# Builds all Docker images with proper tagging

set -e

# Ensure we're in the EquiML root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Building EquiML Docker Images"
echo "=================================="
echo "Working directory: $(pwd)"

# Get version from setup.py
VERSION=$(python -c "exec(open('setup.py').read()); print(globals().get('version', '0.2.0'))" 2>/dev/null || echo "0.2.0")
echo "Building version: $VERSION"

# Verify Dockerfiles exist
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found in $(pwd)"
    echo "Please run this script from the EquiML root directory"
    exit 1
fi

# Build main image
echo ""
echo "Building main EquiML image..."
docker build -t equiml/core:latest -t equiml/core:$VERSION -f Dockerfile .
echo "equiml/core:latest built"

# Build development image
echo ""
echo "Building development image..."
docker build -t equiml/dev:latest -t equiml/dev:$VERSION -f Dockerfile.dev .
echo "equiml/dev:latest built"

# Build production image
echo ""
echo "Building production image..."
docker build -t equiml/prod:latest -t equiml/prod:$VERSION -f Dockerfile.prod .
echo "equiml/prod:latest built"

# Build Jupyter image
echo ""
echo "Building Jupyter image..."
docker build -t equiml/jupyter:latest -t equiml/jupyter:$VERSION -f Dockerfile.jupyter .
echo "equiml/jupyter:latest built"

# Build web demo image
echo ""
echo "Building web demo image..."
if [ -d "examples/web_demo" ]; then
    cd examples/web_demo
    docker build -t equiml/demo:latest -t equiml/demo:$VERSION .
    cd "$PROJECT_ROOT"
    echo "equiml/demo:latest built"
else
    echo "WARNING: examples/web_demo directory not found, skipping demo build"
fi

echo ""
echo " All Docker images built successfully!"
echo ""
echo "Available images:"
docker images | grep equiml

echo ""
echo " Usage examples:"
echo "  Development: docker-compose -f docker-compose.dev.yml up"
echo "  Production:  docker-compose -f docker-compose.prod.yml up"
echo "  Full stack:  docker-compose up"
echo "  Jupyter:     docker run -p 8888:8888 equiml/jupyter"
echo ""
echo " Next steps:"
echo "  1. Test images: ./docker/test.sh"
echo "  2. Push to registry: ./docker/push.sh"
echo "  3. Deploy: docker-compose up -d"