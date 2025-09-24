# EquiML Docker Deployment Guide

**Complete Docker setup for EquiML - from development to production deployment**

## Available Docker Images

EquiML provides multiple Docker images optimized for different use cases:

| Image | Purpose | Size | Use Case |
|-------|---------|------|----------|
| `mkupermann/equiml` | General purpose | ~3GB | CLI usage, scripts |
| `mkupermann/equiml-dev` | Development | ~2.5GB | Development with Jupyter |
| `mkupermann/equiml-prod` | Production | ~1.8GB | Production deployment |
| `mkupermann/equiml-jupyter` | Research | ~6.2GB | Research and notebooks |
| `mkupermann/equiml-demo` | Web demo | ~2.5GB | Live demo application |

## Quick Start

### **Option 1: Try EquiML Instantly**
```bash
# Run EquiML with interactive shell
docker run -it --rm mkupermann/equiml bash

# Inside container:
python -c "from src.data import Data; print('EquiML ready!')"
```

### **Option 2: Run Jupyter Environment**
```bash
# Start Jupyter Lab with EquiML
docker run -p 8888:8888 --rm mkupermann/equiml-jupyter

# Open browser to: http://localhost:8888
# All EquiML examples and tutorials included!
```

### **Option 3: Run Web Demo**
```bash
# Start web demo locally
docker run -p 8501:8501 --rm mkupermann/equiml-demo

# Open browser to: http://localhost:8501
# Upload datasets and get instant bias analysis!
```

## Development Setup

### **Full Development Environment**
```bash
# Clone repository
git clone https://github.com/mkupermann/EquiML.git
cd EquiML

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access services:
# Jupyter Lab: http://localhost:8888
# Streamlit: http://localhost:8501
```

### **Development with Live Reload**
```bash
# Mount local code for live editing
docker run -it --rm \
  -p 8888:8888 \
  -v $(pwd):/app \
  mkupermann/equiml-dev
```

## Production Deployment

### **Option 1: Simple Production**
```bash
# Production deployment with monitoring
docker-compose -f docker-compose.prod.yml up -d

# Services:
# Web App: http://localhost:80
# API: Internal
# Redis Cache: Internal
```

### **Option 2: Full Stack**
```bash
# Complete EquiML stack with monitoring
docker-compose up -d

# Services:
# Web Demo: http://localhost:8502
# Jupyter: http://localhost:8889
# Development: http://localhost:8888
# Monitoring: http://localhost:9090 (Prometheus)
# Dashboards: http://localhost:3000 (Grafana)
```

### **Option 3: Cloud Deployment**
```bash
# AWS ECS
aws ecs create-service --cluster equiml --service-name equiml-web --task-definition equiml:1

# Google Cloud Run
gcloud run deploy equiml --image gcr.io/your-project/mkupermann/equiml-prod:latest

# Azure Container Instances
az container create --resource-group equiml --name equiml --image mkupermann/equiml-prod:latest
```

## Building Images

### **Build All Images**
```bash
# Build all EquiML images
chmod +x docker/build.sh
./docker/build.sh
```

### **Build Individual Images**
```bash
# Core image
docker build -t mkupermann/equiml:latest .

# Development image
docker build -t mkupermann/equiml-dev:latest -f Dockerfile.dev .

# Production image
docker build -t mkupermann/equiml-prod:latest -f Dockerfile.prod .

# Jupyter image
docker build -t mkupermann/equiml-jupyter:latest -f Dockerfile.jupyter .

# Web demo
cd examples/web_demo
docker build -t mkupermann/equiml-demo:latest .
```

## Testing Images

```bash
# Test all images
chmod +x docker/test.sh
./docker/test.sh
```

## Configuration

### **Environment Variables**

#### **Core Configuration**
```bash
# Docker environment variables
PYTHONPATH=/app
EQUIML_ENV=docker|development|production
DEVELOPMENT=0|1
```

#### **Web Demo Configuration**
```bash
# Streamlit configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_THEME_PRIMARY_COLOR=#667eea
```

#### **Database Configuration**
```bash
# PostgreSQL (optional)
POSTGRES_DB=equiml
POSTGRES_USER=equiml
POSTGRES_PASSWORD=your_secure_password

# Redis (optional)
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=your_redis_password
```

### **Volume Mounts**

#### **Data Persistence**
```bash
# Mount data directories
-v ./data:/app/data           # Input datasets
-v ./outputs:/app/outputs     # Generated reports and models
-v ./logs:/app/logs           # Application logs
```

#### **Development Mounts**
```bash
# Live code editing
-v $(pwd):/app                # Full source code
-v ./notebooks:/app/notebooks # Jupyter notebooks
```

## Use Cases

### ** Research & Experimentation**
```bash
# Start Jupyter environment with GPU support
docker run --gpus all -p 8888:8888 \
  -v ./data:/home/jovyan/work/data \
  -v ./experiments:/home/jovyan/work/experiments \
  mkupermann/equiml-jupyter
```

### ** Production API**
```bash
# Production API server
docker run -d --name equiml-api \
  -p 8000:8000 \
  -v ./data:/app/data:ro \
  -v ./outputs:/app/outputs \
  mkupermann/equiml-prod \
  python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### ** Batch Processing**
```bash
# Batch bias analysis
docker run --rm \
  -v ./datasets:/app/data \
  -v ./results:/app/outputs \
  mkupermann/equiml \
  python examples/scripts/quick_bias_check.py /app/data/dataset.csv --target outcome --sensitive gender race
```

### **🌐 Web Demo Deployment**
```bash
# Deploy web demo to cloud
docker run -d --name equiml-demo \
  -p 80:8501 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  mkupermann/equiml-demo
```

## Performance Optimization

### **Memory Optimization**
```bash
# Limit memory usage
docker run --memory=4g --memory-swap=4g mkupermann/equiml

# Use smaller base image for production
# (Already optimized in Dockerfile.prod)
```

### **CPU Optimization**
```bash
# Limit CPU usage
docker run --cpus=2.0 mkupermann/equiml

# Use multi-stage builds for smaller images
# (Implemented in production Dockerfile)
```

### **Network Optimization**
```bash
# Use custom network for better performance
docker network create equiml-net
docker run --network equiml-net mkupermann/equiml
```

## Monitoring & Logging

### **Application Logs**
```bash
# View logs
docker logs equiml-core

# Follow logs in real-time
docker logs -f equiml-core

# Export logs
docker logs equiml-core > equiml.log 2>&1
```

### **Container Monitoring**
```bash
# Monitor resource usage
docker stats

# Monitor specific container
docker stats equiml-core

# Get container info
docker inspect equiml-core
```

### **Health Checks**
```bash
# Check container health
docker ps --filter "name=equiml"

# Manual health check
docker exec equiml-core python -c "import equiml; print('Healthy')"
```

## Troubleshooting

### **Common Issues**

#### **Out of Memory**
```bash
# Increase memory limit
docker run --memory=8g mkupermann/equiml

# Use production image (smaller)
docker run mkupermann/equiml-prod
```

#### **Import Errors**
```bash
# Check Python path
docker exec equiml-core python -c "import sys; print(sys.path)"

# Rebuild image
docker build --no-cache -t mkupermann/equiml .
```

#### **Permission Issues**
```bash
# Run as current user
docker run --user $(id -u):$(id -g) mkupermann/equiml

# Fix volume permissions
sudo chown -R $(id -u):$(id -g) ./data ./outputs
```

### **Debug Mode**
```bash
# Run with debug shell
docker run -it --rm mkupermann/equiml bash

# Check installed packages
docker run --rm mkupermann/equiml pip list

# Verbose build
docker build --progress=plain -t mkupermann/equiml .
```

## Security

### **Production Security**
- Non-root user in production images
- Minimal attack surface (slim base images)
- No unnecessary packages
- Security scanning in CI/CD

### **Secrets Management**
```bash
# Use Docker secrets (Swarm mode)
echo "password" | docker secret create db_password -

# Use environment files
docker run --env-file .env mkupermann/equiml-prod
```

## Registry & Distribution

### **Docker Hub**
```bash
# Tag for Docker Hub
docker tag mkupermann/equiml:latest mkupermann/equiml:latest

# Push to Docker Hub
docker push mkupermann/equiml:latest
```

### **GitHub Container Registry**
```bash
# Tag for GitHub
docker tag mkupermann/equiml:latest ghcr.io/mkupermann/equiml:latest

# Push to GitHub
docker push ghcr.io/mkupermann/equiml:latest
```

### **Private Registry**
```bash
# Tag for private registry
docker tag mkupermann/equiml:latest your-registry.com/equiml:latest

# Push to private registry
docker push your-registry.com/equiml:latest
```

## Automation

### **GitHub Actions Integration**
The provided CI/CD pipeline automatically:
- Builds Docker images on commits
- Tests image functionality
- Pushes to registries on releases
- Validates security

### **Scheduled Builds**
```yaml
# .github/workflows/docker-build.yml
name: Docker Build
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly builds
  release:
    types: [published]
```

## Support

### **Getting Help**
- **Documentation**: [docs/guides/](docs/guides/)
- **Issues**: [GitHub Issues](https://github.com/mkupermann/EquiML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mkupermann/EquiML/discussions)

### **Community**
- Share your Docker configurations
- Contribute improvements
- Report issues and bugs

---

**Docker makes EquiML accessible everywhere - from local development to global cloud deployment!** 