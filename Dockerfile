FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libxkbcommon0 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port and run Streamlit
EXPOSE 8000
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0", "--logger.level=debug"]
