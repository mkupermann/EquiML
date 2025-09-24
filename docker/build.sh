#!/bin/bash
# EquiML Docker Build Script
# Builds all Docker images with proper tagging

set -e

echo " Building EquiML Docker Images"
echo "=================================="

# Get version from setup.py
VERSION=$(python -c "import setup; print(setup.setup.kwargs.get('version', '0.2.0'))" 2>/dev/null || echo "0.2.0")
echo "📦 Building version: $VERSION"

# Build main image
echo ""
echo "🔨 Building main EquiML image..."
docker build -t equiml/core:latest -t equiml/core:$VERSION -f Dockerfile .
echo " equiml/core:latest built"

# Build development image
echo ""
echo "🔨 Building development image..."
docker build -t equiml/dev:latest -t equiml/dev:$VERSION -f Dockerfile.dev .
echo " equiml/dev:latest built"

# Build production image
echo ""
echo "🔨 Building production image..."
docker build -t equiml/prod:latest -t equiml/prod:$VERSION -f Dockerfile.prod .
echo " equiml/prod:latest built"

# Build Jupyter image
echo ""
echo "🔨 Building Jupyter image..."
docker build -t equiml/jupyter:latest -t equiml/jupyter:$VERSION -f Dockerfile.jupyter .
echo " equiml/jupyter:latest built"

# Build web demo image
echo ""
echo "🔨 Building web demo image..."
cd examples/web_demo
docker build -t equiml/demo:latest -t equiml/demo:$VERSION .
cd ../..
echo " equiml/demo:latest built"

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