# EquiML Docker Images

**Public Docker images for EquiML - ready to use worldwide**

## Available Images

All EquiML Docker images are available on both Docker Hub and GitHub Container Registry:

### **Docker Hub (Recommended)**
```bash
# Core EquiML
docker pull mkupermann/equiml:latest

# Web Demo (most popular)
docker pull mkupermann/equiml-demo:latest

# Development Environment
docker pull mkupermann/equiml-dev:latest

# Production Optimized
docker pull mkupermann/equiml-prod:latest

# Jupyter Research Environment
docker pull mkupermann/equiml-jupyter:latest
```

### **GitHub Container Registry**
```bash
# Alternative registry (same images)
docker pull ghcr.io/mkupermann/equiml:latest
docker pull ghcr.io/mkupermann/equiml-demo:latest
```

## Quick Start Commands

### **Try EquiML Instantly**
```bash
# Interactive shell with EquiML
docker run -it --rm mkupermann/equiml:latest bash

# Test bias analysis
docker run --rm mkupermann/equiml:latest python -c "
from src.data import Data
print('EquiML ready for bias analysis!')
"
```

### **Web Demo (Most Popular)**
```bash
# Start web demo
docker run -p 8501:8501 --rm mkupermann/equiml-demo:latest

# Then open: http://localhost:8501
# Upload any CSV dataset for instant bias analysis!
```

### **Jupyter Research Environment**
```bash
# Start Jupyter with EquiML examples
docker run -p 8888:8888 --rm mkupermann/equiml-jupyter:latest

# Then open: http://localhost:8888
# All tutorials and examples pre-loaded!
```

### **Development Environment**
```bash
# Full development setup
docker run -p 8888:8888 -p 8501:8501 \
  -v $(pwd):/app \
  --rm mkupermann/equiml-dev:latest
```

## How Images Are Published

### **Automatic Publishing**
Images are automatically built and published when:
1. **New releases** are tagged on GitHub
2. **Code changes** are pushed to main branch
3. **Docker files** are modified

### **Manual Publishing**
```bash
# Build all images locally
./docker/build.sh

# Push to registries
./docker/push.sh
```

### **Registry URLs**
- **Docker Hub**: https://hub.docker.com/u/mkupermann
- **GitHub Packages**: https://github.com/mkupermann/EquiML/packages

## Image Details

### **mkupermann/equiml-demo:latest**
- **Purpose**: Live web demo for bias analysis
- **Size**: ~1.5GB
- **Port**: 8501 (Streamlit)
- **Features**: Instant bias analysis, certification badges
- **Use**: Public demos, quick testing

### **mkupermann/equiml:latest**
- **Purpose**: Core EquiML functionality
- **Size**: ~2GB
- **Features**: All EquiML modules, CLI tools
- **Use**: Scripting, batch processing, integration

### **mkupermann/equiml-jupyter:latest**
- **Purpose**: Research and education
- **Size**: ~2.5GB
- **Port**: 8888 (Jupyter)
- **Features**: Pre-loaded notebooks, examples
- **Use**: Learning, research, experimentation

### **mkupermann/equiml-dev:latest**
- **Purpose**: Development environment
- **Size**: ~3GB
- **Ports**: 8888 (Jupyter), 8501 (Streamlit)
- **Features**: Development tools, hot reload
- **Use**: Contributing to EquiML, development

### **mkupermann/equiml-prod:latest**
- **Purpose**: Production deployment
- **Size**: ~1GB (optimized)
- **Features**: Minimal, security hardened
- **Use**: Production APIs, cloud deployment

## Update Instructions

### **For Users**
```bash
# Update to latest version
docker pull mkupermann/equiml-demo:latest
docker run -p 8501:8501 --rm mkupermann/equiml-demo:latest
```

### **For Developers**
Images are automatically updated on:
- GitHub releases
- Main branch commits
- Weekly automatic rebuilds

## Security

### **Image Scanning**
All images are scanned for:
- Known vulnerabilities
- Security best practices
- Malware detection

### **Image Signing**
Images are signed with:
- Docker Content Trust
- Cosign signatures (planned)
- SBOM (Software Bill of Materials)

---

**Images are publicly available and ready to use worldwide!**