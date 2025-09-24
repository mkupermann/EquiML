# EquiML - Main Docker Image
# Comprehensive image for running EquiML with all features

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pytest flake8 mypy

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY examples/ ./examples/

# Install EquiML in development mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/outputs /app/data /app/logs

# Copy additional files
COPY README.md CONTRIBUTING.md LICENSE SECURITY.md ./

# Set default command
CMD ["python", "-c", "print('EquiML Docker container ready! Use: docker run -it equiml/core bash')"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import equiml; print('EquiML healthy')" || exit 1

# Labels for metadata
LABEL maintainer="Michael Kupermann <mkupermann@kupermann.com>" \
      description="EquiML - Framework for Equitable and Responsible Machine Learning" \
      version="0.2.0" \
      org.opencontainers.image.source="https://github.com/mkupermann/EquiML" \
      org.opencontainers.image.url="https://github.com/mkupermann/EquiML" \
      org.opencontainers.image.documentation="https://github.com/mkupermann/EquiML/docs/guides/" \
      org.opencontainers.image.title="EquiML" \
      org.opencontainers.image.description="Comprehensive framework for fair and responsible AI development"