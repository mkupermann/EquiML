#!/bin/bash
# EquiML Docker Push Script
# Pushes all images to Docker registries

set -e

echo "Publishing EquiML Docker Images"
echo "==============================="

# Configuration
DOCKER_HUB_USER="mkupermann"
GITHUB_USER="mkupermann"
VERSION=$(python -c "exec(open('setup.py').read()); print(version)" 2>/dev/null || echo "0.2.0")

echo "Version: $VERSION"
echo "Docker Hub user: $DOCKER_HUB_USER"
echo "GitHub user: $GITHUB_USER"

# Login to registries
echo ""
echo "Logging into registries..."

# Docker Hub login
echo "Logging into Docker Hub..."
docker login
echo "✓ Docker Hub login successful"

# GitHub Container Registry login
echo "Logging into GitHub Container Registry..."
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USER --password-stdin 2>/dev/null || echo "⚠️  GitHub login failed (token may not be set)"

echo ""
echo "Building and pushing images..."

# Function to build and push image
push_image() {
    local IMAGE_NAME=$1
    local DOCKERFILE=$2
    local CONTEXT=${3:-.}

    echo ""
    echo "Building and pushing $IMAGE_NAME..."

    # Build image
    docker build -t $IMAGE_NAME:latest -t $IMAGE_NAME:$VERSION -f $DOCKERFILE $CONTEXT

    # Tag for Docker Hub
    docker tag $IMAGE_NAME:latest $DOCKER_HUB_USER/$IMAGE_NAME:latest
    docker tag $IMAGE_NAME:latest $DOCKER_HUB_USER/$IMAGE_NAME:$VERSION

    # Tag for GitHub Container Registry
    docker tag $IMAGE_NAME:latest ghcr.io/$GITHUB_USER/$IMAGE_NAME:latest
    docker tag $IMAGE_NAME:latest ghcr.io/$GITHUB_USER/$IMAGE_NAME:$VERSION

    # Push to Docker Hub
    echo "Pushing to Docker Hub..."
    docker push $DOCKER_HUB_USER/$IMAGE_NAME:latest
    docker push $DOCKER_HUB_USER/$IMAGE_NAME:$VERSION

    # Push to GitHub Container Registry
    echo "Pushing to GitHub Container Registry..."
    docker push ghcr.io/$GITHUB_USER/$IMAGE_NAME:latest || echo "GitHub push failed (check token)"
    docker push ghcr.io/$GITHUB_USER/$IMAGE_NAME:$VERSION || echo "GitHub push failed (check token)"

    echo "✓ $IMAGE_NAME published successfully"
}

# Push all images
push_image "equiml" "Dockerfile" "."
push_image "equiml-dev" "Dockerfile.dev" "."
push_image "equiml-prod" "Dockerfile.prod" "."
push_image "equiml-jupyter" "Dockerfile.jupyter" "."
push_image "equiml-demo" "Dockerfile" "examples/web_demo"

echo ""
echo "🎉 All images published successfully!"
echo ""
echo "Available on Docker Hub:"
echo "  docker pull $DOCKER_HUB_USER/equiml:latest"
echo "  docker pull $DOCKER_HUB_USER/equiml-dev:latest"
echo "  docker pull $DOCKER_HUB_USER/equiml-prod:latest"
echo "  docker pull $DOCKER_HUB_USER/equiml-jupyter:latest"
echo "  docker pull $DOCKER_HUB_USER/equiml-demo:latest"
echo ""
echo "Available on GitHub Container Registry:"
echo "  docker pull ghcr.io/$GITHUB_USER/equiml:latest"
echo "  docker pull ghcr.io/$GITHUB_USER/equiml-demo:latest"
echo ""
echo "Public usage:"
echo "  docker run -p 8501:8501 $DOCKER_HUB_USER/equiml-demo"
echo "  docker run -p 8888:8888 $DOCKER_HUB_USER/equiml-jupyter"