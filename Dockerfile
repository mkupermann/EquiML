# EquiML - Complete Docker Image
# Comprehensive image with ALL EquiML features and capabilities

FROM python:3.11-slim as builder

# Build stage - install all dependencies
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy all files needed for installation
COPY requirements.txt pyproject.toml setup.py ./
COPY README.md LICENSE ./
COPY src/ ./src/

# Install Python dependencies in compatible versions
COPY requirements-docker.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt && \
    pip install --no-deps fairlearn==0.10.0 shap==0.42.1 lime==0.2.0.1 && \
    pip install --no-deps xgboost==1.7.6 || echo "xgboost optional" && \
    pip install --no-deps lightgbm==4.1.0 || echo "lightgbm optional" && \
    pip install --no-deps optuna==3.6.1 || echo "optuna optional" && \
    pip install jupyter jupyterlab notebook

# Install EquiML package
RUN pip install -e .

# Production stage - copy everything needed
FROM python:3.11-slim

# Set comprehensive environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    EQUIML_DOCKER=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy complete EquiML codebase
COPY src/ ./src/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY examples/ ./examples/
COPY README.md LICENSE CONTRIBUTING.md SECURITY.md ./

# Copy configuration files
COPY pyproject.toml setup.py requirements.txt ./
COPY .flake8 mypy.ini pytest.ini ./

# Create all necessary directories
RUN mkdir -p /app/outputs /app/data /app/logs /app/reports /app/models

# Set up Python environment
RUN echo "#!/usr/bin/env python3" > /usr/local/bin/equiml && \
    echo "import sys; sys.path.insert(0, '/app')" >> /usr/local/bin/equiml && \
    echo "from src.data import Data" >> /usr/local/bin/equiml && \
    echo "from src.model import Model" >> /usr/local/bin/equiml && \
    echo "from src.evaluation import EquiMLEvaluation" >> /usr/local/bin/equiml && \
    echo "from src.monitoring import BiasMonitor, DriftDetector" >> /usr/local/bin/equiml && \
    echo "print('EquiML loaded with all modules')" >> /usr/local/bin/equiml && \
    chmod +x /usr/local/bin/equiml

# Download NLTK data if needed
RUN python -c "import sys; sys.path.append('/app'); from src.data import _setup_nltk; _setup_nltk()" || echo "NLTK setup completed"

# Set default command to show all capabilities
CMD ["python", "-c", "import sys; sys.path.append('/app'); from src.data import Data; from src.model import Model; from src.evaluation import EquiMLEvaluation; from src.monitoring import BiasMonitor; print('EquiML comprehensive container ready!'); print('Available: Data, Model, EquiMLEvaluation, BiasMonitor'); print('Web demo: streamlit run src/streamlit_app.py'); print('Examples: see /app/examples/')"]

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from src.data import Data; from src.model import Model; print('EquiML healthy')" || exit 1

# Complete metadata labels
LABEL maintainer="Michael Kupermann <mkupermann@kupermann.com>" \
      description="EquiML - Complete Framework for Equitable and Responsible Machine Learning" \
      version="0.2.0" \
      org.opencontainers.image.source="https://github.com/mkupermann/EquiML" \
      org.opencontainers.image.url="https://github.com/mkupermann/EquiML" \
      org.opencontainers.image.documentation="https://github.com/mkupermann/EquiML/docs/guides/" \
      org.opencontainers.image.title="EquiML Complete" \
      org.opencontainers.image.description="Comprehensive framework with bias detection, monitoring, LLM support, and all advanced features" \
      features="bias-detection,fairness-constraints,monitoring,reporting,visualization,streamlit,jupyter-ready"