#!/usr/bin/env python3
"""
Extract WavLM weights from microsoft/wavlm-base-plus and upload to
kimballwightman/SalesS2S-AudioEncoder repo.

This simplified version directly saves the microsoft WavLM model without
requiring custom model classes.
"""

import torch
import os
import sys
from transformers import Wav2Vec2Model, Wav2Vec2Config
from huggingface_hub import HfApi
import json

print("🔧 Extracting WavLM and uploading to SalesS2S-AudioEncoder")
print("=" * 70)

# Check for HF token
token = os.environ.get("HF_TOKEN")
if not token:
    print("❌ No HF_TOKEN found. Set with: export HF_TOKEN=your_token")
    sys.exit(1)

# Step 1: Load pretrained microsoft WavLM
print("\n📥 Step 1: Loading microsoft/wavlm-base-plus...")
try:
    wavlm_model = Wav2Vec2Model.from_pretrained(
        "microsoft/wavlm-base-plus",
        torch_dtype=torch.bfloat16
    )
    wavlm_config = Wav2Vec2Config.from_pretrained("microsoft/wavlm-base-plus")
    print("✅ WavLM loaded")
except Exception as e:
    print(f"❌ Failed to load WavLM: {e}")
    sys.exit(1)

# Step 2: Save to local directory
print("\n💾 Step 2: Saving WavLM weights and configs...")
output_dir = "./wavlm_encoder_extracted"
os.makedirs(output_dir, exist_ok=True)

try:
    # Save WavLM model weights
    wavlm_model.save_pretrained(output_dir)

    # Save main config (transformers config)
    wavlm_config.save_pretrained(output_dir)

    # Save WavLM-specific audio_encoder_config.json for OmniSpeech
    audio_encoder_config = {
        "model_type": "wavlm",
        "hidden_size": 768,
        "d_model": 1280,  # For compatibility with adapter
        "num_hidden_layers": 12,
        "max_source_positions": 1500,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "activation_function": "gelu",
        "dropout": 0.1,
        "attention_dropout": 0.1
    }
    with open(os.path.join(output_dir, "audio_encoder_config.json"), "w") as f:
        json.dump(audio_encoder_config, f, indent=2)

    print(f"✅ Saved to {output_dir}")
    print(f"   Files created:")
    print(f"   - pytorch_model.bin (WavLM weights)")
    print(f"   - config.json (transformers config)")
    print(f"   - audio_encoder_config.json (OmniSpeech config)")
except Exception as e:
    print(f"❌ Failed to save: {e}")
    sys.exit(1)

# Step 3: Upload to HuggingFace
print("\n☁️  Step 3: Uploading to kimballwightman/SalesS2S-AudioEncoder...")
try:
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        repo_id="kimballwightman/SalesS2S-AudioEncoder",
        repo_type="model",
        token=token
    )
    print("✅ Upload complete!")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("🎉 EXTRACTION COMPLETE!")
print("=" * 70)
print(f"\n📦 kimballwightman/SalesS2S-AudioEncoder now contains:")
print("   - pytorch_model.bin: WavLM weights (768 hidden, 12 layers)")
print("   - config.json: Transformers Wav2Vec2Config")
print("   - audio_encoder_config.json: OmniSpeech-specific config")
print("   - Replaces previous Qwen2AudioEncoder weights")
print("\n📋 Next steps:")
print("   1. ✅ WavLM weights uploaded to AudioEncoder repo")
print("   2. 🔨 Custom OpenS2S code will load from this repo")
print("   3. 🚀 Test with: cd ~/OpenS2S && python3 startup.py")
print("\n" + "=" * 70)
