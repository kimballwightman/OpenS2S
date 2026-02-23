#!/usr/bin/env python3
"""
GPTQ Quantization Script for OmniSpeech LLM
============================================

This script extracts the Qwen LLM from the unified OmniSpeech checkpoint
and quantizes it to 8-bit using AutoGPTQ for persistent on-disk quantization.

Requirements:
    pip install auto-gptq

Usage:
    python3 quantize_llm_gptq.py

Expected output:
    - Quantized LLM saved to /models/OpenS2S-llm-gptq/ (~8-10GB)
    - Faster loading: 30-60s vs 2-5min
    - Memory savings: ~12-14GB total model size vs 21GB
"""

import os
import sys
import torch
import shutil

# Add src to path
sys.path.insert(0, '/app')
from src.modeling_omnispeech import OmniSpeechModel

SOURCE_PATH = "/models/OpenS2S"
LLM_TEMP_PATH = "/models/OpenS2S-llm-temp"
LLM_GPTQ_PATH = "/models/OpenS2S-llm-gptq"

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

    # Copy tokenizer files from source to temp directory
    print()
    print(f"Copying tokenizer files from {SOURCE_PATH} to {LLM_TEMP_PATH}...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "added_tokens.json"
    ]

    for file in tokenizer_files:
        src = os.path.join(SOURCE_PATH, file)
        dst = os.path.join(LLM_TEMP_PATH, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"   Copied {file}")

    print("✅ Standalone LLM checkpoint saved with tokenizer")
    print()

    return LLM_TEMP_PATH

def quantize_with_gptq(llm_checkpoint_path):
    """Quantize LLM using AutoGPTQ."""
    print("=" * 70)
    print("🔧 Step 2/3: Quantizing LLM with AutoGPTQ (8-bit)")
    print("=" * 70)
    print()

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ Error: AutoGPTQ not installed")
        print()
        print("Install with:")
        print("    pip install auto-gptq")
        print()
        sys.exit(1)

    print("Loading LLM for quantization...")
    print("This may take 1-2 minutes...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        llm_checkpoint_path,
        trust_remote_code=True
    )

    print("✅ Tokenizer loaded")
    print()

    # Quantization config for 8-bit
    quantize_config = BaseQuantizeConfig(
        bits=8,
        group_size=128,
        desc_act=False,  # Disable activation ordering for faster quantization
        damp_percent=0.01
    )

    print("Loading model with quantization config...")
    # Load model with quantization config
    model = AutoGPTQForCausalLM.from_pretrained(
        llm_checkpoint_path,
        quantize_config=quantize_config,
        device_map="auto"  # Will use GPU for quantization
    )

    print("✅ LLM loaded for quantization")
    print()

    print("Quantizing to 8-bit GPTQ...")
    print("This may take 5-10 minutes...")
    print()

    # Prepare calibration data (simple approach - a few sample texts)
    calibration_texts = [
        "Hello, how are you today?",
        "I would like to learn more about artificial intelligence.",
        "The weather is nice today.",
        "Can you help me with a programming question?",
        "What is the capital of France?"
    ]

    # Tokenize calibration data
    examples = []
    for text in calibration_texts:
        example = tokenizer(text, return_tensors="pt")
        examples.append(example)

    print("📊 Quantizing with calibration data...")
    # Quantize the model
    model.quantize(examples)

    print("✅ LLM quantized to 8-bit")
    print()

    # Save quantized model
    print(f"Saving quantized LLM to {LLM_GPTQ_PATH}...")
    os.makedirs(LLM_GPTQ_PATH, exist_ok=True)
    model.save_quantized(LLM_GPTQ_PATH, use_safetensors=True)
    tokenizer.save_pretrained(LLM_GPTQ_PATH)

    print("✅ Quantized LLM saved")
    print()

    return LLM_GPTQ_PATH

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
    for root, dirs, files in os.walk(LLM_GPTQ_PATH):
        for f in files:
            quantized_size += os.path.getsize(os.path.join(root, f))

    print(f"Original checkpoint (full): {original_size / 1e9:.2f} GB")
    print(f"Quantized LLM only: {quantized_size / 1e9:.2f} GB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    print()

    if quantized_size > 12e9:
        print("⚠️  WARNING: Quantized size larger than expected")
        print("   Expected: ~8-10GB for 8-bit quantized LLM")
        print(f"   Got: {quantized_size / 1e9:.2f}GB")
    else:
        print("✅ Quantization successful!")
        print(f"   Size reduction: {original_size / 1e9 - quantized_size / 1e9:.1f}GB saved")

    print()

def cleanup_temp():
    """Cleanup temporary files."""
    if os.path.exists(LLM_TEMP_PATH):
        print(f"Cleaning up temporary files at {LLM_TEMP_PATH}...")
        shutil.rmtree(LLM_TEMP_PATH)
        print("✅ Cleanup complete")

def main():
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║   GPTQ Quantization for OmniSpeech LLM (Persistent 8-bit)    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Step 1: Extract LLM from unified checkpoint
        llm_checkpoint = extract_llm_checkpoint()

        # Step 2: Quantize with GPTQ
        quantized_path = quantize_with_gptq(llm_checkpoint)

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
        print("1. Quantized LLM saved to:", LLM_GPTQ_PATH)
        print("2. Restart container to use quantized model")
        print("3. Expected memory usage: ~12-14GB (vs 21GB original)")
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
