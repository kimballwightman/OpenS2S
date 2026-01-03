# OpenS2S Inference Server - GPU VM Dockerfile
# Optimized for g2-standard-4 with NVIDIA L4, CUDA 12.2

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python

# Copy requirements first for better Docker caching
COPY requirements.txt /app/

# Upgrade pip and setuptools for compilation
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2)
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies that flash_attn needs
RUN python3.11 -m pip install --no-cache-dir \
    transformers==4.51.0 \
    accelerate \
    numpy==2.3.1 \
    einops

# Install packages that require compilation (skip flash_attn for now - optimization only)
RUN python3.11 -m pip install --no-cache-dir \
    natten==0.20.1 || echo "Natten install failed, continuing without it"

# Install remaining dependencies
RUN python3.11 -m pip install --no-cache-dir \
    optimum \
    datasets \
    diffusers \
    peft \
    librosa==0.10.1 \
    soundfile==0.13.1 \
    scipy \
    inflect \
    phonemizer \
    Unidecode \
    HyperPyYAML==1.2.2 \
    fastapi==0.116.0 \
    uvicorn==0.35.0 \
    aiofiles \
    pydantic

# Install remaining OpenS2S requirements (skip already installed packages)
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt || true

# Copy OpenS2S source code
COPY . /app/

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:/app

# Expose inference server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set proper permissions
RUN chmod +x /app/model_worker.py

# Run original OpenS2S model worker (for testing original implementation)
CMD ["python3.11", "model_worker.py", "--host", "0.0.0.0", "--port", "8000"]