# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
COPY ml/requirements_ml.txt ./ml/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r ml/requirements_ml.txt

# Copy application files
COPY *.py .
COPY monitor_config.json* ./

# Create static directory and copy files
COPY static/ ./static/

# Copy ML models directory
COPY ml/ ./ml/

# Create directory for data persistence
RUN mkdir -p /app/data

# Expose API server port
EXPOSE 8000

# Health check for API server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/api/status || exit 1

# Run API server with uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
