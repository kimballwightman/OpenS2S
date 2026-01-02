#!/bin/bash
# GPU VM Cleanup Script - Free up disk space before Docker build
# Run this on the GPU VM if you encounter "no space left on device" errors

echo "🧹 GPU VM Disk Cleanup Script"
echo "This will free up disk space for Docker builds"
echo ""

# Function to show disk usage
show_disk_usage() {
    echo "📊 Current disk usage:"
    df -h / | grep -E "(Filesystem|/dev/)"
    echo ""
}

# Show initial disk usage
echo "Before cleanup:"
show_disk_usage

# Stop all running containers
echo "🛑 Stopping all Docker containers..."
docker stop $(docker ps -q) 2>/dev/null || true

# Clean up Docker system completely
echo "🐳 Cleaning up Docker system..."
docker system prune -a -f --volumes
docker builder prune -a -f

# Clean up containerd (if using containerd)
echo "📦 Cleaning up containerd..."
sudo systemctl stop containerd 2>/dev/null || true
sudo rm -rf /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/* 2>/dev/null || true
sudo systemctl start containerd 2>/dev/null || true

# Clean up system package cache
echo "📦 Cleaning up system packages..."
sudo apt-get clean
sudo apt-get autoremove -y
sudo apt-get autoclean

# Clean up logs
echo "📝 Cleaning up system logs..."
sudo journalctl --vacuum-time=1d
sudo find /var/log -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true

# Clean up temporary files
echo "🗑️  Cleaning up temporary files..."
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*
sudo rm -rf ~/.cache/*

# Clean up pip cache
echo "🐍 Cleaning up Python pip cache..."
pip3 cache purge 2>/dev/null || true

# Clean up any old kernels (Ubuntu/Debian)
echo "🔧 Cleaning up old kernels..."
sudo apt-get autoremove --purge -y 2>/dev/null || true

# Show final disk usage
echo ""
echo "After cleanup:"
show_disk_usage

# Check if we have enough space now
AVAILABLE_GB=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
echo "💾 Available space: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -gt 15 ]; then
    echo "✅ Good! You have enough space for Docker build"
elif [ "$AVAILABLE_GB" -gt 10 ]; then
    echo "⚠️  Sufficient space, but build may be tight. Monitor during build."
else
    echo "❌ Still low on space. Consider:"
    echo "   - Expanding the VM disk size"
    echo "   - Removing unnecessary files manually"
    echo "   - Using a VM with larger disk"
fi

echo ""
echo "🎯 Cleanup complete! You can now run: ./build_and_run.sh"