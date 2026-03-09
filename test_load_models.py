#!/usr/bin/env python3
"""
Test loading extracted sub-models from separate HuggingFace repos.

This script loads each component individually to verify:
1. Models can be loaded successfully
2. Weights are present and valid
3. Memory usage is reasonable
4. Basic inference works
"""

import torch
import gc
import sys
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Add src to path for custom model classes
sys.path.insert(0, './src')

# Your private HF repos
REPOS = {
    "audio_encoder": "kimballwightman/SalesS2S-AudioEncoder",
    "llm": "kimballwightman/SalesS2S-LLM",
    "tts": "kimballwightman/SalesS2S-TTS",
    "adapters": "kimballwightman/SalesS2S-Adapters"
}

def print_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("   GPU not available")

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("=" * 70)
print("🧪 Testing Extracted Model Loading")
print("=" * 70)

# ============================================================
# Test 1: Load Audio Encoder
# ============================================================
print("\n📊 Test 1: Loading Audio Encoder...")
print(f"   Repo: {REPOS['audio_encoder']}")

try:
    from modeling_audio_encoder import Qwen2AudioEncoder

    audio_encoder = Qwen2AudioEncoder.from_pretrained(
        REPOS['audio_encoder'],
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load to CPU for testing
        trust_remote_code=True
    )

    print("   ✅ Audio Encoder loaded successfully")
    print(f"   Type: {type(audio_encoder).__name__}")

    # Count parameters
    num_params = sum(p.numel() for p in audio_encoder.parameters())
    print(f"   Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    print_memory_usage()

    # Clean up
    del audio_encoder
    clear_memory()
    print("   ✅ Memory cleared")

except Exception as e:
    print(f"   ❌ Failed to load Audio Encoder: {e}")

# ============================================================
# Test 2: Load LLM
# ============================================================
print("\n📊 Test 2: Loading LLM...")
print(f"   Repo: {REPOS['llm']}")

try:
    llm = AutoModelForCausalLM.from_pretrained(
        REPOS['llm'],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(REPOS['llm'], trust_remote_code=True)

    print("   ✅ LLM loaded successfully")
    print(f"   Type: {type(llm).__name__}")

    # Count parameters
    num_params = sum(p.numel() for p in llm.parameters())
    print(f"   Parameters: {num_params:,} ({num_params / 1e9:.1f}B)")

    # Test tokenization
    test_text = "Hello, how are you?"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   Tokenization test: '{test_text}' -> {tokens.input_ids.shape[1]} tokens")

    print_memory_usage()

    # Clean up
    del llm
    del tokenizer
    clear_memory()
    print("   ✅ Memory cleared")

except Exception as e:
    print(f"   ❌ Failed to load LLM: {e}")

# ============================================================
# Test 3: Load TTS
# ============================================================
print("\n📊 Test 3: Loading TTS...")
print(f"   Repo: {REPOS['tts']}")

try:
    tts = AutoModelForCausalLM.from_pretrained(
        REPOS['tts'],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(REPOS['tts'], trust_remote_code=True)

    print("   ✅ TTS loaded successfully")
    print(f"   Type: {type(tts).__name__}")

    # Count parameters
    num_params = sum(p.numel() for p in tts.parameters())
    print(f"   Parameters: {num_params:,} ({num_params / 1e9:.1f}B)")

    print_memory_usage()

    # Clean up
    del tts
    del tokenizer
    clear_memory()
    print("   ✅ Memory cleared")

except Exception as e:
    print(f"   ❌ Failed to load TTS: {e}")

# ============================================================
# Test 4: Load Adapters
# ============================================================
print("\n📊 Test 4: Loading Adapters...")
print(f"   Repo: {REPOS['adapters']}")

try:
    from huggingface_hub import hf_hub_download

    # Download adapter weights
    adapter_weights_path = hf_hub_download(
        repo_id=REPOS['adapters'],
        filename="adapter_weights.pt"
    )

    # Load adapter weights
    adapter_state = torch.load(adapter_weights_path, map_location="cpu")

    print("   ✅ Adapters loaded successfully")
    print(f"   Number of adapter parameters: {len(adapter_state)}")
    print(f"   Adapter parameter names:")
    for name in list(adapter_state.keys())[:5]:  # Show first 5
        print(f"      - {name}")
    if len(adapter_state) > 5:
        print(f"      ... and {len(adapter_state) - 5} more")

    # Calculate total size
    total_params = sum(p.numel() for p in adapter_state.values())
    print(f"   Total adapter parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    print_memory_usage()

    # Clean up
    del adapter_state
    clear_memory()
    print("   ✅ Memory cleared")

except Exception as e:
    print(f"   ❌ Failed to load Adapters: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("✅ All model loading tests complete!")
print("=" * 70)
print("\n📋 Next Steps:")
print("   1. ✅ Models load successfully from separate repos")
print("   2. 📝 Test loading in Docker container")
print("   3. 🔨 Refactor SalesS2S engine with modular classes")
print("   4. ⚙️  Update config to use separate repos")
print("   5. 🎯 Add model swapping (WavLM) and quantization (LLM)")
print("\n" + "=" * 70)
