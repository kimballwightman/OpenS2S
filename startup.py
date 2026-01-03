#!/usr/bin/env python3
"""
OpenS2S Startup Script
Downloads required HuggingFace models at runtime if they don't exist.
Models are stored in persistent /models directory (mounted volume).
"""

import os
import sys
import subprocess
from pathlib import Path

# Model configuration
MODELS_DIR = "/models"
MODELS_CONFIG = [
    {
        "repo_id": "CASIA-LM/OpenS2S",
        "local_path": f"{MODELS_DIR}/OpenS2S",
        "description": "OpenS2S main model"
    },
    {
        "repo_id": "THUDM/glm-4-voice-decoder",
        "local_path": f"{MODELS_DIR}/glm-4-voice-decoder",
        "description": "GLM-4 Voice Decoder"
    }
]

def download_model_if_needed(repo_id, local_path, description):
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

        # Download model
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )

        print(f"✅ Successfully downloaded {description}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return False

def ensure_models_available():
    """Ensure all required models are downloaded and available."""

    print("🚀 OpenS2S Startup - Checking required models...")
    print(f"📁 Models directory: {MODELS_DIR}")

    # Create models directory if it doesn't exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    success_count = 0
    for model_config in MODELS_CONFIG:
        if download_model_if_needed(**model_config):
            success_count += 1

    if success_count == len(MODELS_CONFIG):
        print(f"✅ All {len(MODELS_CONFIG)} models are ready!")
        return True
    else:
        print(f"❌ Only {success_count}/{len(MODELS_CONFIG)} models are available")
        return False

def start_model_worker():
    """Start the OpenS2S model worker with correct model paths."""

    print("🎯 Starting OpenS2S model worker...")

    # Model paths for runtime
    opens2s_model_path = f"{MODELS_DIR}/OpenS2S"
    decoder_model_path = f"{MODELS_DIR}/glm-4-voice-decoder"

    # Build command arguments
    cmd = [
        "python3", "model_worker.py",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model-path", opens2s_model_path,
        "--flow-path", decoder_model_path,
        "--no-register"  # Skip controller registration for standalone mode
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
    try:
        # Step 1: Ensure models are downloaded
        if not ensure_models_available():
            print("❌ Cannot start without required models")
            sys.exit(1)

        # Step 2: Start inference server
        start_model_worker()

    except KeyboardInterrupt:
        print("\n👋 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        sys.exit(1)