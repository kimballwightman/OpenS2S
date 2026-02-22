#!/bin/bash
# OpenS2S Inference Server - Build and Run Script
# For NVIDIA L4 GPU VM (g2-standard-4)
# Uses WavLM encoder + 8-bit quantized LLM for optimized VRAM usage

echo "🚀 OpenS2S Docker Build & Run Script (WavLM + 8-bit Quantization)"
echo "GPU VM: g2-standard-4 with NVIDIA L4, CUDA 12.2"
echo "Expected VRAM: ~8-10GB (down from ~20GB with Whisper)"
echo ""

# Check if we're in the right directory
if [ ! -f "model_worker.py" ]; then
    echo "❌ Error: Must run from OpenS2S directory containing model_worker.py"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA Docker runtime not working. Check GPU setup."
    exit 1
fi

echo "✅ Docker and NVIDIA runtime verified"

# Build the Docker image
echo "📦 Building OpenS2S inference server image..."
docker build -t opens2s:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully"

# Stop any existing container
echo "🛑 Stopping any existing containers..."
docker stop opens2s-server 2>/dev/null || true
docker rm opens2s-server 2>/dev/null || true

# Run the container with WavLM encoder
echo "🎯 Starting OpenS2S inference server with WavLM + 8-bit quantization..."
docker run -d \
    --name opens2s-server \
    --gpus all \
    --restart unless-stopped \
    -p 8000:8000 \
    -p 21001:21001 \
    -v /tmp:/tmp \
    --shm-size=2g \
    opens2s:latest \
    python3 model_worker.py \
      --host 0.0.0.0 \
      --port 8000 \
      --model-path /models/OpenS2S \
      --flow-path /models/glm-4-voice-decoder \
      --controller-address http://localhost:21001 \
      --worker-address http://localhost:8000 \
      --audio-processor wavlm

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

echo "✅ OpenS2S inference server started successfully!"
echo ""
echo "📋 Server Information:"
echo "   - Container: opens2s-server"
echo "   - Model Worker Port: 8000"
echo "   - Controller Port: 21001"
echo "   - Audio Encoder: WavLM (replaces Whisper, saves 3-5GB VRAM)"
echo "   - LLM Quantization: 8-bit (saves 8-10GB VRAM)"
echo "   - Health check: http://localhost:8000/health"
echo "   - WebSocket: ws://localhost:8000/ws/stream"
echo ""
echo "📊 Useful commands:"
echo "   docker logs -f opens2s-server          # View logs (watch for quantization messages)"
echo "   docker exec -it opens2s-server bash    # Shell access"
echo "   docker exec -it opens2s-server nvidia-smi    # Check GPU memory (should be ~8-10GB)"
echo "   docker stop opens2s-server             # Stop server"
echo "   docker restart opens2s-server          # Restart server"
echo ""

# Wait a moment and check health
echo "🩺 Checking server health..."
sleep 5

if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ Server is healthy and responding!"
else
    echo "⚠️ Server may still be starting up. Check logs: docker logs opens2s-server"
fi

echo ""
echo "🎉 OpenS2S inference server is ready!"
echo "💡 Backend should connect to: http://$(curl -s ifconfig.me):8000"