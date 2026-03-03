#!/usr/bin/env python3
"""
OpenS2S Startup Script
Downloads required HuggingFace models at runtime if they don't exist.
Models are stored in persistent /models directory (mounted volume).

HuggingFace Authentication:
- Checks for HF_ACCESS_TOKEN environment variable
- If not found, attempts to fetch from GCP Secret Manager
- Token required for private repository access
"""

import os
import sys
import subprocess
from pathlib import Path

# Model configuration
MODELS_DIR = "/models"
MODELS_CONFIG = [
    {
        "repo_id": "kimballwightman/SalesS2S",
        "local_path": f"{MODELS_DIR}/SalesS2S",
        "description": "OpenS2S main model"
    },
    {
        "repo_id": "kimballwightman/SalesS2S-Voice-Decoder",
        "local_path": f"{MODELS_DIR}/SalesS2S-Voice-Decoder",
        "description": "GLM-4 Voice Decoder"
    }
]

def get_hf_token():
    """Get HuggingFace token from HF_TOKEN environment variable."""
    token = os.environ.get("HF_TOKEN")
    if token:
        print("🔑 Using HF_TOKEN from environment variable")
        return token
    else:
        print("⚠️  No HF_TOKEN found - can only access public repos")
        return None


def download_model_if_needed(repo_id, local_path, description, hf_token=None):
    """Download HuggingFace model if it doesn't already exist."""

    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"✅ {description} already exists at {local_path}")
        return True

    print(f"📥 Downloading {description} from {repo_id}...")
    print(f"   Target: {local_path}")

    try:
        from huggingface_hub import snapshot_download

        # Create parent directory if needed
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Download model with token for private repo access
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            token=hf_token  # Pass token for authentication
        )

        print(f"✅ Successfully downloaded {description}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        if "401" in str(e) or "403" in str(e):
            print("   💡 This is a private repo - ensure HF_TOKEN environment variable is set")
        return False

def ensure_models_available():
    """Ensure all required models are downloaded and available."""

    print("🚀 OpenS2S Startup - Checking required models...")
    print(f"📁 Models directory: {MODELS_DIR}")

    # Create models directory if it doesn't exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Get HuggingFace token from environment
    hf_token = get_hf_token()

    success_count = 0
    for model_config in MODELS_CONFIG:
        if download_model_if_needed(**model_config, hf_token=hf_token):
            success_count += 1

    if success_count == len(MODELS_CONFIG):
        print(f"✅ All {len(MODELS_CONFIG)} models are ready!")
        return True
    else:
        print(f"❌ Only {success_count}/{len(MODELS_CONFIG)} models are available")
        return False

def start_controller():
    """Start the controller in the background."""

    print("🎮 Starting controller...")

    # Start controller in background
    controller_process = subprocess.Popen(
        ["python3", "controller.py", "--host", "0.0.0.0", "--port", "21001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait a bit for controller to start
    import time
    time.sleep(3)

    # Check if controller is still running
    if controller_process.poll() is not None:
        print("❌ Controller failed to start")
        sys.exit(1)

    print("✅ Controller started on port 21001")
    return controller_process

def start_model_worker():
    """Start the OpenS2S model worker with correct model paths."""

    print("🎯 Starting OpenS2S model worker...")

    # Model paths for runtime - using private repos
    opens2s_model_path = f"{MODELS_DIR}/SalesS2S"
    decoder_model_path = f"{MODELS_DIR}/SalesS2S-Voice-Decoder"

    print(f"   Using OpenS2S model: {opens2s_model_path}")
    print(f"   Using Voice Decoder: {decoder_model_path}")

    # Build command arguments - REMOVED --no-register so worker connects to controller
    # Note: WavLM is now hardcoded in model config, no need for --audio-processor flag
    cmd = [
        "python3", "model_worker.py",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model-path", opens2s_model_path,
        "--flow-path", decoder_model_path,
        "--controller-address", "http://localhost:21001",
        "--worker-address", "http://localhost:8000"
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    print("📡 Server will be available at http://0.0.0.0:8000")
    print("🏥 Health check: http://0.0.0.0:8000/health")

    # Execute model worker
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Model worker failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    controller_process = None
    try:
        # Step 1: Ensure models are downloaded
        if not ensure_models_available():
            print("❌ Cannot start without required models")
            sys.exit(1)

        # Step 2: Start controller
        controller_process = start_controller()

        # Step 3: Start model worker (will register with controller)
        start_model_worker()

    except KeyboardInterrupt:
        print("\n👋 Startup interrupted by user")
        if controller_process:
            controller_process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        if controller_process:
            controller_process.terminate()
        sys.exit(1)