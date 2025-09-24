#!/bin/bash
# EquiML Web Demo Deployment Script

set -e

echo "🚀 EquiML Web Demo Deployment"
echo "============================="

# Configuration
DOMAIN="equiml.ai"
APP_DIR="/opt/equiml-demo"
SERVICE_NAME="equiml-demo"

echo "📦 Building Docker containers..."
docker-compose build

echo "🔧 Setting up SSL certificates..."
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificate for testing (replace with real certificates)
if [ ! -f "ssl/${DOMAIN}.crt" ]; then
    echo "Generating self-signed SSL certificate for testing..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/${DOMAIN}.key \
        -out ssl/${DOMAIN}.crt \
        -subj "/C=US/ST=State/L=City/O=EquiML/CN=${DOMAIN}"

    echo "⚠️  WARNING: Using self-signed certificate for testing only!"
    echo "   For production, replace with real SSL certificates from Let's Encrypt or your provider"
fi

echo "🌐 Starting services..."
docker-compose up -d

echo "🔍 Checking service health..."
sleep 10

# Health check
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ EquiML demo is running successfully!"
    echo ""
    echo "🌍 Access your demo at:"
    echo "   Local: http://localhost:8501"
    echo "   Production: https://${DOMAIN}"
    echo ""
    echo "📊 Monitor with:"
    echo "   docker-compose logs -f"
    echo ""
    echo "🛑 Stop with:"
    echo "   docker-compose down"
else
    echo "❌ Health check failed. Check logs:"
    docker-compose logs
    exit 1
fi

echo ""
echo "🎉 EquiML Web Demo deployment completed!"
echo "Ready to attract global community attention!"