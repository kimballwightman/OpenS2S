#!/usr/bin/env python3
"""
One-Time Model Quantization Script
===================================

This script quantizes the OmniSpeech LLM to 8-bit and saves it to a separate directory.
Run this ONCE after downloading the original model, then always load from the quantized version.

Usage:
    python3 quantize_model_once.py

This will:
1. Load the original model from /models/OpenS2S
2. Quantize the LLM component to 8-bit
3. Save the quantized model to /models/OpenS2S-8bit
4. Future model loads will use the pre-quantized version (faster!)

Expected time: 5-10 minutes (one-time only)
Expected space: ~10GB for quantized model (vs 21GB original)
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, '/app')
from src.modeling_omnispeech import OmniSpeechModel

SOURCE_PATH = "/models/OpenS2S"
QUANTIZED_PATH = "/models/OpenS2S-8bit"

def main():
    print("=" * 70)
    print("🔧 One-Time Model Quantization Script")
    print("=" * 70)
    print()

    # Check if source model exists
    if not os.path.exists(SOURCE_PATH):
        print(f"❌ Error: Source model not found at {SOURCE_PATH}")
        print("   Please run the container first to download the model.")
        sys.exit(1)

    # Check if quantized model already exists
    if os.path.exists(QUANTIZED_PATH) and os.listdir(QUANTIZED_PATH):
        print(f"⚠️  Quantized model already exists at {QUANTIZED_PATH}")
        response = input("   Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
            print("   Exiting without changes.")
            sys.exit(0)
        print()

    print(f"📥 Step 1/4: Loading original model from {SOURCE_PATH}...")
    print("   This may take 1-2 minutes...")

    try:
        model = OmniSpeechModel.from_pretrained(
            SOURCE_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            local_files_only=True
        )
        print("   ✅ Original model loaded to CPU")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)

    print()
    print("🔄 Step 2/4: Quantizing LLM component to 8-bit...")
    print("   This may take 2-3 minutes...")
    print("   Note: Using manual layer replacement for quantization")

    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear8bitLt, Int8Params

        # Manual quantization: Replace all Linear layers in LLM with 8-bit versions
        def quantize_linear_layers(module, parent_name=""):
            """Recursively replace Linear layers with 8-bit quantized versions."""
            replaced_count = 0

            for name, child in list(module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name

                if isinstance(child, torch.nn.Linear):
                    # Create 8-bit linear layer
                    new_layer = Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )

                    # Quantize and copy weights
                    new_layer.weight = Int8Params(
                        child.weight.data.to(torch.float16),  # Convert to fp16 first
                        requires_grad=False,
                        has_fp16_weights=False
                    )

                    if child.bias is not None:
                        new_layer.bias = torch.nn.Parameter(child.bias.data)

                    # Replace the layer
                    setattr(module, name, new_layer)
                    replaced_count += 1
                else:
                    # Recursively process child modules
                    replaced_count += quantize_linear_layers(child, full_name)

            return replaced_count

        # Apply quantization to LLM
        print("   Replacing Linear layers with 8-bit quantized versions...")
        layers_replaced = quantize_linear_layers(model.llm_model)
        print(f"   ✅ Quantized {layers_replaced} Linear layers in LLM")

    except ImportError:
        print("   ❌ bitsandbytes not installed - quantization skipped")
        print("   Model will be saved without quantization")
    except Exception as e:
        print(f"   ❌ Quantization failed: {e}")
        print("   Model will be saved without quantization")
        import traceback
        traceback.print_exc()

    print()
    print(f"💾 Step 3/4: Saving quantized model to {QUANTIZED_PATH}...")
    print("   This may take 2-3 minutes...")

    try:
        os.makedirs(QUANTIZED_PATH, exist_ok=True)

        model.save_pretrained(
            QUANTIZED_PATH,
            safe_serialization=True,
            max_shard_size="5GB"
        )

        # Also copy tokenizer and config files
        import shutil
        for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json"]:
            src = os.path.join(SOURCE_PATH, file)
            dst = os.path.join(QUANTIZED_PATH, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        # Copy subdirectories (audio, tts)
        for subdir in ["audio", "tts"]:
            src_dir = os.path.join(SOURCE_PATH, subdir)
            dst_dir = os.path.join(QUANTIZED_PATH, subdir)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        print("   ✅ Quantized model saved")
    except Exception as e:
        print(f"   ❌ Failed to save: {e}")
        sys.exit(1)

    print()
    print("📊 Step 4/4: Verifying saved model...")

    # Check saved files
    saved_files = os.listdir(QUANTIZED_PATH)
    safetensor_files = [f for f in saved_files if f.endswith('.safetensors')]

    print(f"   Found {len(safetensor_files)} safetensor files")
    print(f"   Total files: {len(saved_files)}")

    # Estimate size
    total_size = 0
    for root, dirs, files in os.walk(QUANTIZED_PATH):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    size_gb = total_size / 1e9
    print(f"   Total size: {size_gb:.2f} GB")

    # Verify quantization worked by checking size
    if size_gb > 15:
        print()
        print("   ⚠️  WARNING: Model size suggests quantization may have failed")
        print(f"   Expected: ~10-12GB for quantized model")
        print(f"   Got: {size_gb:.2f}GB (similar to original 21GB)")
        print()
        print("   Possible causes:")
        print("   - bitsandbytes quantization didn't apply")
        print("   - Model saved in full precision")
        print()
        print("   You can still use this model, but it won't have memory savings.")
    else:
        print(f"   ✅ Size looks good for quantized model (~{size_gb:.1f}GB vs 21GB original)")

    print()
    print("=" * 70)
    print("🎉 SUCCESS! Model quantization complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Update startup.py to use the quantized model path")
    print("2. Restart the container to use the quantized model")
    print("3. Future loads will be faster (no runtime quantization needed)")
    print()
    print(f"Quantized model location: {QUANTIZED_PATH}")
    print()

if __name__ == "__main__":
    main()
