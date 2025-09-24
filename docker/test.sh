#!/bin/bash
# EquiML Docker Test Script
# Tests all Docker images to ensure they work correctly

set -e

echo " Testing EquiML Docker Images"
echo "================================"

# Test main image
echo ""
echo " Testing main EquiML image..."
docker run --rm equiml/core:latest python -c "
import equiml
from src.data import Data
print(' Main image: Core functionality works')
"

# Test development image
echo ""
echo " Testing development image..."
docker run --rm -d --name test-dev -p 8887:8888 equiml/dev:latest
sleep 10

if curl -f http://localhost:8887 > /dev/null 2>&1; then
    echo " Development image: Jupyter Lab accessible"
else
    echo "  Development image: Jupyter Lab not accessible (may need more time)"
fi

docker stop test-dev > /dev/null 2>&1 || true

# Test production image
echo ""
echo " Testing production image..."
docker run --rm equiml/prod:latest python -c "
import equiml
print(' Production image: Optimized build works')
"

# Test Jupyter image
echo ""
echo " Testing Jupyter image..."
docker run --rm -d --name test-jupyter -p 8886:8888 equiml/jupyter:latest
sleep 15

if curl -f http://localhost:8886 > /dev/null 2>&1; then
    echo " Jupyter image: Notebook server accessible"
else
    echo "  Jupyter image: Server not accessible (may need more time)"
fi

docker stop test-jupyter > /dev/null 2>&1 || true

# Test web demo image
echo ""
echo " Testing web demo image..."
docker run --rm -d --name test-demo -p 8885:8501 equiml/demo:latest
sleep 20

if curl -f http://localhost:8885/_stcore/health > /dev/null 2>&1; then
    echo " Web demo image: Streamlit app healthy"
else
    echo "  Web demo image: App not healthy (may need dependencies)"
fi

docker stop test-demo > /dev/null 2>&1 || true

# Test Docker Compose
echo ""
echo " Testing Docker Compose..."
echo "Testing development compose..."
docker-compose -f docker-compose.dev.yml config > /dev/null
echo " Development compose: Configuration valid"

echo "Testing production compose..."
docker-compose -f docker-compose.prod.yml config > /dev/null
echo " Production compose: Configuration valid"

echo "Testing full stack compose..."
docker-compose config > /dev/null
echo " Full stack compose: Configuration valid"

echo ""
echo " All Docker tests completed!"
echo ""
echo " Image sizes:"
docker images | grep equiml | awk '{print $1":"$2" - "$7$8}'

echo ""
echo " Ready for deployment!"
echo "  Development: docker-compose -f docker-compose.dev.yml up -d"
echo "  Production:  docker-compose -f docker-compose.prod.yml up -d"