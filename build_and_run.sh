#!/bin/bash
# OpenS2S Inference Server - Build and Run Script
# For NVIDIA L4 GPU VM (g2-standard-4)

echo "🚀 OpenS2S Docker Build & Run Script"
echo "GPU VM: g2-standard-4 with NVIDIA L4, CUDA 12.2"
echo ""

# Check if we're in the right directory
if [ ! -f "inference_server.py" ]; then
    echo "❌ Error: Must run from OpenS2S directory containing inference_server.py"
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
docker build -t opens2s-inference:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully"

# Stop any existing container
echo "🛑 Stopping any existing containers..."
docker stop opens2s-server 2>/dev/null || true
docker rm opens2s-server 2>/dev/null || true

# Run the container
echo "🎯 Starting OpenS2S inference server..."
docker run -d \
    --name opens2s-server \
    --gpus all \
    --restart unless-stopped \
    -p 8000:8000 \
    -v /tmp:/tmp \
    --shm-size=2g \
    opens2s-inference:latest

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

echo "✅ OpenS2S inference server started successfully!"
echo ""
echo "📋 Server Information:"
echo "   - Container: opens2s-server"
echo "   - Port: 8000"
echo "   - Health check: http://localhost:8000/health"
echo "   - Inference endpoint: http://localhost:8000/stream_infer"
echo ""
echo "📊 Useful commands:"
echo "   docker logs -f opens2s-server    # View logs"
echo "   docker exec -it opens2s-server bash    # Shell access"
echo "   docker stop opens2s-server       # Stop server"
echo "   docker restart opens2s-server    # Restart server"
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