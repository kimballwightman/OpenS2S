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
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    && python3.11 -m ensurepip \
    && python3.11 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Copy requirements first for better Docker caching
COPY requirements.txt /app/

# Note: pip already upgraded during Python 3.11 installation above

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2)
RUN pip3 install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies that flash_attn needs
RUN pip3 install --no-cache-dir \
    transformers==4.51.0 \
    accelerate \
    numpy==2.3.1 \
    einops

# Install packages that require compilation (skip flash_attn for now - optimization only)
RUN pip3 install --no-cache-dir \
    natten==0.20.1 || echo "Natten install failed, continuing without it"

# Install critical dependencies explicitly (common import failures)
RUN pip3 install --no-cache-dir \
    HyperPyYAML==1.1.0 \
    omegaconf==2.3.0 \
    conformer==0.3.2 \
    einops==0.8.1 \
    fire==0.7.0 \
    rich==14.0.0 \
    tqdm==4.66.5 \
    librosa==0.10.1 \
    soundfile==0.13.1 \
    inflect \
    phonemizer==3.3.0 \
    Unidecode==1.3.8

# Install remaining dependencies
RUN pip3 install --no-cache-dir \
    optimum \
    datasets \
    diffusers \
    peft==0.4.0 \
    librosa==0.10.1 \
    soundfile==0.13.1 \
    scipy \
    inflect \
    phonemizer \
    Unidecode \
    fastapi==0.116.0 \
    uvicorn==0.35.0 \
    aiofiles \
    pydantic

# Install remaining OpenS2S requirements (skip already installed packages)
RUN pip3 install --no-cache-dir -r requirements.txt || true

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
RUN chmod +x /app/model_worker.py /app/startup.py

# Create models directory for volume mount
RUN mkdir -p /models

# Run startup script that downloads models and starts server
CMD ["python3", "startup.py"]