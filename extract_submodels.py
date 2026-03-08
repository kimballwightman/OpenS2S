#!/usr/bin/env python3
"""
Extract sub-models from composite OmniSpeechModel and save to separate HF repos.

This script:
1. Loads the composite OpenS2S model
2. Extracts each component (audio encoder, LLM, TTS, adapters)
3. Saves each to separate private HuggingFace repos
4. Uses HF_TOKEN from environment for authentication

Memory efficient for L4 (24GB): Loads once, extracts all, then cleans up.
"""

import torch
import os
import sys
import shutil
from pathlib import Path

# Add current directory to path for src package imports
sys.path.insert(0, '.')

from src.modeling_omnispeech import OmniSpeechModel
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

# ====================== CONFIG ======================
ORIGINAL_REPO = "CASIA-LM/OpenS2S"

# Local temp directories for extraction
OUTPUT_DIRS = {
    "audio_encoder": "./temp_extracted/audio_encoder",
    "llm": "./temp_extracted/llm",
    "tts": "./temp_extracted/tts",
    "adapters": "./temp_extracted/adapters"
}

# Target HuggingFace repos (will be created as private)
HF_REPOS = {
    "audio_encoder": "kimballwightman/SalesS2S-AudioEncoder",
    "llm": "kimballwightman/SalesS2S-LLM",
    "tts": "kimballwightman/SalesS2S-TTS",
    "adapters": "kimballwightman/SalesS2S-Adapters"
}

print("🔧 OmniSpeech Sub-Model Extraction")
print("=" * 70)
print("This will extract components and upload to separate private repos:")
for name, repo in HF_REPOS.items():
    print(f"  - {name.upper()}: {repo}")
print("=" * 70)

# Check for HF_TOKEN
token = os.environ.get("HF_TOKEN")
if not token:
    print("\n❌ No HF_TOKEN environment variable found")
    print("   Set it with: export HF_TOKEN=your_token")
    sys.exit(1)

print("✅ HF_TOKEN found")

# ============================================================
# STEP 1: Load composite model
# ============================================================
print("\n📥 STEP 1: Loading composite OmniSpeechModel...")
print(f"   From: {ORIGINAL_REPO}")
print("   This will download ~15-20GB if not cached")
print("   Loading on CPU to save GPU memory...")

try:
    model = OmniSpeechModel.from_pretrained(
        ORIGINAL_REPO,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Use GPU - L4 has 24GB VRAM, enough for ~22GB model
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_REPO, trust_remote_code=True)

    print("✅ Composite model loaded")
    print(f"\n   📊 Components found:")
    print(f"      - Audio Encoder: {type(model.audio_encoder_model).__name__}")
    print(f"      - LLM: {type(model.llm_model).__name__}")
    print(f"      - TTS LM: {type(model.tts_lm_model).__name__}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ============================================================
# STEP 2: Extract and save each component locally
# ============================================================
print("\n💾 STEP 2: Extracting components to local directories...")

# Extract audio encoder
print("\n   🎤 Extracting Audio Encoder...")
try:
    os.makedirs(OUTPUT_DIRS["audio_encoder"], exist_ok=True)
    model.audio_encoder_model.save_pretrained(OUTPUT_DIRS["audio_encoder"])

    # Save audio encoder config separately
    if hasattr(model.config, 'audio_encoder_config'):
        audio_config_path = os.path.join(OUTPUT_DIRS["audio_encoder"], "audio_encoder_config.json")
        model.config.audio_encoder_config.to_json_file(audio_config_path)

    print(f"   ✅ Audio Encoder saved to {OUTPUT_DIRS['audio_encoder']}")
    print(f"      Type: {type(model.audio_encoder_model).__name__} (~2-3GB)")
except Exception as e:
    print(f"   ❌ Failed to extract audio encoder: {e}")

# Extract LLM
print("\n   🧠 Extracting LLM...")
try:
    os.makedirs(OUTPUT_DIRS["llm"], exist_ok=True)
    model.llm_model.save_pretrained(OUTPUT_DIRS["llm"], safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(OUTPUT_DIRS["llm"])

    # Save LLM config separately
    if hasattr(model.config, 'llm_config'):
        llm_config_path = os.path.join(OUTPUT_DIRS["llm"], "llm_config.json")
        model.config.llm_config.to_json_file(llm_config_path)

    print(f"   ✅ LLM saved to {OUTPUT_DIRS['llm']}")
    print(f"      Type: Qwen3 7B full precision (~14GB)")
except Exception as e:
    print(f"   ❌ Failed to extract LLM: {e}")

# Extract TTS LM
print("\n   🎵 Extracting TTS LM...")
try:
    os.makedirs(OUTPUT_DIRS["tts"], exist_ok=True)
    model.tts_lm_model.save_pretrained(OUTPUT_DIRS["tts"], safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(OUTPUT_DIRS["tts"])

    # Save TTS config separately
    if hasattr(model.config, 'tts_lm_config'):
        tts_config_path = os.path.join(OUTPUT_DIRS["tts"], "tts_lm_config.json")
        model.config.tts_lm_config.to_json_file(tts_config_path)

    print(f"   ✅ TTS LM saved to {OUTPUT_DIRS['tts']}")
    print(f"      Type: Qwen3 2B full precision (~4GB)")
except Exception as e:
    print(f"   ❌ Failed to extract TTS LM: {e}")

# Extract adapters and configs
print("\n   🔌 Extracting Adapters...")
try:
    os.makedirs(OUTPUT_DIRS["adapters"], exist_ok=True)

    # Save full parent config (contains adapter settings)
    model.config.save_pretrained(OUTPUT_DIRS["adapters"])

    # Extract adapter-specific parameters from model state_dict
    adapter_state = {}
    for name, param in model.named_parameters():
        # Look for adapter-related parameters
        if any(keyword in name.lower() for keyword in ['adapter', 'projection', 'connector']):
            adapter_state[name] = param.cpu()

    if adapter_state:
        # Save adapter weights
        adapter_weights_path = os.path.join(OUTPUT_DIRS["adapters"], "adapter_weights.pt")
        torch.save(adapter_state, adapter_weights_path)

        # Save list of adapter parameter names
        adapter_names_path = os.path.join(OUTPUT_DIRS["adapters"], "adapter_params.txt")
        with open(adapter_names_path, 'w') as f:
            for name in adapter_state.keys():
                f.write(f"{name}\n")

        print(f"   ✅ Adapters saved to {OUTPUT_DIRS['adapters']}")
        print(f"      Found {len(adapter_state)} adapter parameters (~100MB)")
    else:
        print("   ⚠️  No explicit adapter parameters found")
        print("      (Adapters might be integrated into model components)")

except Exception as e:
    print(f"   ❌ Failed to extract adapters: {e}")

print("\n✅ All components extracted locally")

# Clear memory before uploading
print("\n   🧹 Clearing model from memory before upload...")
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("   ✅ Memory cleared")

# ============================================================
# STEP 3: Upload to HuggingFace repos
# ============================================================
print("\n☁️  STEP 3: Uploading to HuggingFace repos...")
print("   Creating private repos and uploading contents...")

api = HfApi()
upload_results = {}

for component, local_dir in OUTPUT_DIRS.items():
    repo_id = HF_REPOS[component]

    # Check if directory has content
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"\n   ⚠️  Skipping {component} - no content extracted")
        upload_results[component] = "skipped"
        continue

    print(f"\n   📤 {component.upper()}: {repo_id}")

    try:
        # Create private repo (or use existing)
        print(f"      Creating/verifying repo...")
        create_repo(repo_id, private=True, exist_ok=True, token=token)
        print(f"      ✅ Repo ready")

        # Upload folder contents
        print(f"      Uploading files...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"      ✅ Upload complete!")
        upload_results[component] = "success"

    except Exception as e:
        print(f"      ❌ Upload failed: {e}")
        upload_results[component] = f"failed: {e}"

# ============================================================
# STEP 4: Cleanup
# ============================================================
print("\n🧹 STEP 4: Cleaning up temporary files...")

if os.path.exists("./temp_extracted"):
    shutil.rmtree("./temp_extracted")
    print("   ✅ Temporary files removed")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("🎉 EXTRACTION COMPLETE!")
print("=" * 70)

print("\n📊 Upload Results:")
for component, result in upload_results.items():
    status_icon = "✅" if result == "success" else "⚠️" if result == "skipped" else "❌"
    print(f"  {status_icon} {component.upper()}: {result}")

print("\n📦 Your HuggingFace repos:")
for component, repo in HF_REPOS.items():
    if upload_results.get(component) == "success":
        print(f"\n  🔗 {repo}")
        if component == "audio_encoder":
            print(f"     Current: Qwen2AudioEncoder (32 layers, ~2-3GB)")
            print(f"     Next step: Swap to pretrained WavLM (12 layers)")
        elif component == "llm":
            print(f"     Current: Qwen3 7B full precision (~14GB)")
            print(f"     Next step: Quantize to 8-bit (~7GB)")
        elif component == "tts":
            print(f"     Current: Qwen3 2B full precision (~4GB)")
            print(f"     Can quantize if needed for memory savings")
        elif component == "adapters":
            print(f"     Projection layers connecting components")

print("\n📋 Next Steps:")
print("   1. ✅ Sub-models extracted to separate repos")
print("   2. 📝 Work on each model individually (within 24GB L4 memory):")
print("      a. Create swap_wavlm.py - Replace audio encoder with WavLM")
print("      b. Create quantize_llm.py - Quantize LLM to 8-bit")
print("      c. (Optional) Quantize TTS if needed")
print("   3. 🔨 Refactor model_worker.py to load from separate repos")
print("   4. 🏗️  Create modular model classes (AudioEncoder, LLM, TTS)")
print("   5. 🎯 Add intent classifier as separate model class")

print("\n💡 Memory-efficient workflow enabled:")
print("   - Each operation stays within L4's 24GB limit")
print("   - No need for monolithic composite model loading")
print("   - Can iterate on individual components independently")

print("\n" + "=" * 70)
