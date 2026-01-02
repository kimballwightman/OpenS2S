# OpenS2S Inference Server - GPU VM Dockerfile
# Optimized for g2-standard-4 with NVIDIA L4, CUDA 12.2

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first for better Docker caching
COPY requirements.txt /app/

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2)
RUN pip3 install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other ML dependencies
RUN pip3 install --no-cache-dir \
    transformers==4.51.0 \
    accelerate \
    optimum \
    datasets \
    diffusers \
    peft

# Install audio processing dependencies
RUN pip3 install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.13.1 \
    scipy \
    numpy \
    einops \
    inflect \
    phonemizer \
    Unidecode

# Install web server dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.116.0 \
    uvicorn==0.35.0 \
    aiofiles \
    pydantic

# Install remaining OpenS2S requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy OpenS2S source code
COPY . /app/

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:/app:$PYTHONPATH

# Expose inference server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set proper permissions
RUN chmod +x /app/inference_server.py

# Run inference server
CMD ["python3", "inference_server.py"]