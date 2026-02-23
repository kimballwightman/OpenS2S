#!/usr/bin/env python3
"""
AWQ Quantization Script for OmniSpeech LLM
==========================================

This script extracts the Qwen LLM from the unified OmniSpeech checkpoint
and quantizes it to 4-bit using AutoAWQ for persistent on-disk quantization.

Requirements:
    pip install autoawq

Usage:
    python3 quantize_llm_awq.py

Expected output:
    - Quantized LLM saved to /models/OpenS2S-llm-awq/ (~4-5GB)
    - Faster loading: 30-60s vs 2-5min
    - Memory savings: ~10GB total model size vs 21GB
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, '/app')
from src.modeling_omnispeech import OmniSpeechModel

SOURCE_PATH = "/models/OpenS2S"
LLM_TEMP_PATH = "/models/OpenS2S-llm-temp"
LLM_AWQ_PATH = "/models/OpenS2S-llm-awq"

def extract_llm_checkpoint():
    """Extract LLM weights from unified OmniSpeech checkpoint."""
    print("=" * 70)
    print("📦 Step 1/3: Extracting LLM from OmniSpeech Checkpoint")
    print("=" * 70)
    print()

    if not os.path.exists(SOURCE_PATH):
        print(f"❌ Error: Source model not found at {SOURCE_PATH}")
        sys.exit(1)

    print(f"Loading OmniSpeech model from {SOURCE_PATH}...")
    print("This may take 1-2 minutes...")

    # Load full model to CPU
    model = OmniSpeechModel.from_pretrained(
        SOURCE_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        local_files_only=True
    )

    print("✅ OmniSpeech model loaded")
    print()

    # Extract LLM
    print("Extracting LLM component...")
    llm_model = model.llm_model
    llm_config = model.llm_config

    # Get LLM info
    total_params = sum(p.numel() for p in llm_model.parameters())
    print(f"LLM Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"LLM Type: {type(llm_model).__name__}")
    print(f"LLM Config: {llm_config.model_type}")

    # Save LLM as standalone checkpoint
    print()
    print(f"Saving standalone LLM checkpoint to {LLM_TEMP_PATH}...")
    os.makedirs(LLM_TEMP_PATH, exist_ok=True)

    llm_model.save_pretrained(
        LLM_TEMP_PATH,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    print("✅ Standalone LLM checkpoint saved")
    print()

    return LLM_TEMP_PATH

def quantize_with_awq(llm_checkpoint_path):
    """Quantize LLM using AutoAWQ."""
    print("=" * 70)
    print("🔧 Step 2/3: Quantizing LLM with AutoAWQ")
    print("=" * 70)
    print()

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ Error: AutoAWQ not installed")
        print()
        print("Install with:")
        print("    pip install autoawq")
        print()
        sys.exit(1)

    print("Loading LLM for quantization...")
    print("This may take 1-2 minutes...")

    # Load model and tokenizer
    model = AutoAWQForCausalLM.from_pretrained(
        llm_checkpoint_path,
        device_map="cpu",
        safetensors=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        llm_checkpoint_path,
        trust_remote_code=True
    )

    print("✅ LLM loaded for quantization")
    print()

    # Quantization config
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    print("Quantizing to 4-bit AWQ...")
    print("This may take 3-5 minutes...")
    print()

    # Note: AWQ requires calibration data for best results
    # For now, we'll use a simple quantization without calibration
    # For production, you'd want to provide representative samples

    model.quantize(tokenizer, quant_config=quant_config)

    print("✅ LLM quantized to 4-bit")
    print()

    # Save quantized model
    print(f"Saving quantized LLM to {LLM_AWQ_PATH}...")
    model.save_quantized(LLM_AWQ_PATH, safetensors=True)
    tokenizer.save_pretrained(LLM_AWQ_PATH)

    print("✅ Quantized LLM saved")
    print()

    return LLM_AWQ_PATH

def verify_quantization():
    """Verify the quantized model."""
    print("=" * 70)
    print("📊 Step 3/3: Verifying Quantization")
    print("=" * 70)
    print()

    # Check sizes
    original_size = 0
    for root, dirs, files in os.walk(SOURCE_PATH):
        for f in files:
            if f.endswith('.safetensors') or f.endswith('.bin'):
                original_size += os.path.getsize(os.path.join(root, f))

    quantized_size = 0
    for root, dirs, files in os.walk(LLM_AWQ_PATH):
        for f in files:
            quantized_size += os.path.getsize(os.path.join(root, f))

    print(f"Original checkpoint (full): {original_size / 1e9:.2f} GB")
    print(f"Quantized LLM only: {quantized_size / 1e9:.2f} GB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    print()

    if quantized_size > 10e9:
        print("⚠️  WARNING: Quantized size larger than expected")
        print("   Expected: ~4-5GB for 4-bit quantized LLM")
        print(f"   Got: {quantized_size / 1e9:.2f}GB")
    else:
        print("✅ Quantization successful!")
        print(f"   Size reduction: {original_size / 1e9 - quantized_size / 1e9:.1f}GB saved")

    print()

def cleanup_temp():
    """Cleanup temporary files."""
    import shutil
    if os.path.exists(LLM_TEMP_PATH):
        print(f"Cleaning up temporary files at {LLM_TEMP_PATH}...")
        shutil.rmtree(LLM_TEMP_PATH)
        print("✅ Cleanup complete")

def main():
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║   AWQ Quantization for OmniSpeech LLM (Persistent on Disk)    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Step 1: Extract LLM from unified checkpoint
        llm_checkpoint = extract_llm_checkpoint()

        # Step 2: Quantize with AWQ
        quantized_path = quantize_with_awq(llm_checkpoint)

        # Step 3: Verify
        verify_quantization()

        # Cleanup
        cleanup_temp()

        print()
        print("=" * 70)
        print("🎉 SUCCESS! LLM Quantization Complete")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Quantized LLM saved to:", LLM_AWQ_PATH)
        print("2. Update model_worker.py to load from quantized path")
        print("3. Restart container to use quantized model")
        print("4. Expected memory usage: ~10-12GB (vs 21GB original)")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR: Quantization Failed")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
