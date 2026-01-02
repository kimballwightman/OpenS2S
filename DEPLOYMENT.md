# OpenS2S Inference Server Deployment

## 🚀 Quick Start (GPU VM)

### 1. SSH to GPU VM
```bash
ssh dev-gpu
```

### 2. Pull Latest Changes
```bash
cd ~/OpenS2S
git pull origin main
```

### 3. Build and Run
```bash
./build_and_run.sh
```

### 4. Verify Deployment
```bash
# Check health
curl http://localhost:8000/health

# Check logs
docker logs -f opens2s-server
```

## 🏗️ VM Configuration

**Instance:** `opens2s-dev-gpu`
- **Type:** g2-standard-4 (4 vCPUs, 16 GB RAM)
- **GPU:** NVIDIA L4 (23 GB memory)
- **CUDA:** 12.2, Driver 535.247.01
- **External IP:** 34.169.169.71
- **SSH:** `ssh dev-gpu`

## 🐳 Docker Commands

```bash
# Build image
docker build -t opens2s-inference:latest .

# Run with GPU support
docker run -d \
  --name opens2s-server \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  --shm-size=2g \
  opens2s-inference:latest

# View logs
docker logs -f opens2s-server

# Shell access
docker exec -it opens2s-server bash

# Stop/restart
docker stop opens2s-server
docker restart opens2s-server

# Remove container
docker rm -f opens2s-server
```

## 🔌 API Endpoints

### Health Check
```bash
GET http://34.169.169.71:8000/health
```

### HTTP/1.1 Streaming Inference
```bash
POST http://34.169.169.71:8000/stream_infer
Content-Type: application/octet-stream

# Streams raw PCM binary data
# Frame format: [4 bytes length][PCM16 payload]
```

## 🧪 Testing

### Basic Health Test
```bash
curl -f http://34.169.169.71:8000/health
```

### Backend Connection Test
Update your local `.env` file:
```env
GCP_GPU_INSTANCE_URL=http://34.169.169.71:8000
```

Then run your backend locally:
```bash
docker-compose up --build
```

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check logs
docker logs opens2s-server

# Rebuild image
docker build --no-cache -t opens2s-inference:latest .
```

### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Restart with more shared memory
docker run -d --name opens2s-server --gpus all --shm-size=4g -p 8000:8000 opens2s-inference:latest
```

### Port Issues
```bash
# Check what's using port 8000
sudo netstat -tlnp | grep :8000

# Kill existing processes
sudo pkill -f "port 8000"
```

## 📊 Monitoring

### Check GPU Usage
```bash
watch nvidia-smi
```

### Check Container Resources
```bash
docker stats opens2s-server
```

### View Real-time Logs
```bash
docker logs -f opens2s-server | grep -E "(🎵|✅|❌|🔊)"
```

## 🔄 Development Workflow

1. **Make changes** to code locally
2. **Commit and push** to GitHub
3. **SSH to VM**: `ssh dev-gpu`
4. **Pull changes**: `cd ~/OpenS2S && git pull`
5. **Rebuild and run**: `./build_and_run.sh`
6. **Test**: Backend → VM → Response pipeline

## 🌐 Network Configuration

- **VM Internal:** Container runs on port 8000
- **VM External:** Accessible at `34.169.169.71:8000`
- **Firewall:** Port 8000 open for HTTP traffic
- **Backend Config:** Set `GCP_GPU_INSTANCE_URL=http://34.169.169.71:8000`