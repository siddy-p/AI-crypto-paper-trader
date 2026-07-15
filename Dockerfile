# Use a slim Debian-based image which provides pre-compiled wheels for numpy/pandas/xgboost
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies needed for compiling if any packages fall back
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY main.py .
COPY src/ ./src/

# Expose web dashboard port
EXPOSE 5001

# Run the entry point
CMD ["python", "main.py"]
