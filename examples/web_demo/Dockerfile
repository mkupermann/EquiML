FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy EquiML source from parent directories
COPY ../../src ./src
COPY ../../tests ./tests

# Copy web demo application files
COPY app.py analytics.py landing_page.py ./

# Set Python path
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/outputs /app/data

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]