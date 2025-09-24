#!/bin/bash
# Simplified Docker Hub Publishing Script for mkupermann account

set -e

DOCKER_USER="mkupermann"
VERSION="0.2.0"

echo "Publishing EquiML to Docker Hub ($DOCKER_USER)"
echo "=============================================="

# Check if logged in
if ! docker info | grep -q "Username: $DOCKER_USER"; then
    echo "Please login to Docker Hub first:"
    echo "docker login"
    echo "Username: $DOCKER_USER"
    echo "Password: [your Docker Hub password]"
    exit 1
fi

echo "✓ Logged in to Docker Hub as $DOCKER_USER"

# Build images if they don't exist locally
echo ""
echo "Building images locally..."

# Build main EquiML image
if ! docker images | grep -q "equiml/core"; then
    echo "Building core image..."
    docker build -t equiml/core:latest .
fi

# Build web demo image
if ! docker images | grep -q "equiml/demo"; then
    echo "Building demo image..."
    cd examples/web_demo
    docker build -t equiml/demo:latest .
    cd ../..
fi

# Build development image
if ! docker images | grep -q "equiml/dev"; then
    echo "Building dev image..."
    docker build -t equiml/dev:latest -f Dockerfile.dev .
fi

# Build production image
if ! docker images | grep -q "equiml/prod"; then
    echo "Building prod image..."
    docker build -t equiml/prod:latest -f Dockerfile.prod .
fi

# Build Jupyter image
if ! docker images | grep -q "equiml/jupyter"; then
    echo "Building jupyter image..."
    docker build -t equiml/jupyter:latest -f Dockerfile.jupyter .
fi

echo "✓ All images built locally"

# Function to tag and push image
publish_image() {
    local LOCAL_NAME=$1
    local DOCKER_HUB_NAME=$2

    echo ""
    echo "Publishing $DOCKER_HUB_NAME..."

    # Tag for Docker Hub
    docker tag $LOCAL_NAME:latest $DOCKER_USER/$DOCKER_HUB_NAME:latest
    docker tag $LOCAL_NAME:latest $DOCKER_USER/$DOCKER_HUB_NAME:$VERSION

    # Push to Docker Hub
    echo "Pushing $DOCKER_HUB_NAME:latest..."
    docker push $DOCKER_USER/$DOCKER_HUB_NAME:latest

    echo "Pushing $DOCKER_HUB_NAME:$VERSION..."
    docker push $DOCKER_USER/$DOCKER_HUB_NAME:$VERSION

    echo "✓ $DOCKER_HUB_NAME published successfully"
}

# Publish all images
echo ""
echo "Publishing to Docker Hub..."

publish_image "equiml/core" "equiml"
publish_image "equiml/demo" "equiml-demo"
publish_image "equiml/dev" "equiml-dev"
publish_image "equiml/prod" "equiml-prod"
publish_image "equiml/jupyter" "equiml-jupyter"

echo ""
echo "🎉 All EquiML images published to Docker Hub!"
echo ""
echo "Public usage commands:"
echo "======================"
echo ""
echo "# Web demo (most popular)"
echo "docker run -p 8501:8501 --rm $DOCKER_USER/equiml-demo:latest"
echo ""
echo "# Jupyter environment"
echo "docker run -p 8888:8888 --rm $DOCKER_USER/equiml-jupyter:latest"
echo ""
echo "# Core EquiML"
echo "docker run -it --rm $DOCKER_USER/equiml:latest bash"
echo ""
echo "# Development environment"
echo "docker run -p 8888:8888 -p 8501:8501 --rm $DOCKER_USER/equiml-dev:latest"
echo ""
echo "# Production deployment"
echo "docker run -p 8000:8000 --rm $DOCKER_USER/equiml-prod:latest"
echo ""
echo "View on Docker Hub:"
echo "https://hub.docker.com/u/$DOCKER_USER"
echo ""
echo "Update README.md with these working commands!"