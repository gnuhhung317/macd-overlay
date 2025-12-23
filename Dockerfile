# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY data_processor.py .
COPY telegram_notifier.py .
COPY streamlit_monitor.py .
COPY config.py .

# Create directory for Streamlit config
RUN mkdir -p ~/.streamlit

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_monitor.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
