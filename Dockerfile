FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep image lightweight and fast on Render/cloud
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install backend requirements
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/app ./app
COPY backend/models ./models
COPY frontend ./frontend

# Create uploads directory
RUN mkdir -p uploads

# Expose default port
EXPOSE 8000

# Run uvicorn binding to 0.0.0.0 and dynamic $PORT for Render / cloud hosts
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
