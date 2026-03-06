#!/usr/bin/env python3
"""
Bake script for SalesS2S:
1. Load original OpenS2S (with Qwen2AudioEncoder to avoid weight warnings)
2. Replace audio encoder with pretrained WavLM
3. Quantize the LLM to 8-bit with AutoGPTQ
4. Save complete model with WavLM + quantized LLM
5. Also save quantized LLM separately for runtime loading pattern
6. Upload both to private HuggingFace repos

Result: Both repos contain quantized models
"""

import torch
import os
import sys
import shutil
from pathlib import Path

# Add OpenS2S src to path
sys.path.insert(0, './src')

from modeling_omnispeech import OmniSpeechModel
from transformers import AutoTokenizer, WavLMModel, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from huggingface_hub import HfApi, create_repo

# ====================== CONFIG ======================
ORIGINAL_REPO = "CASIA-LM/OpenS2S"
OUTPUT_MAIN = "./SalesS2S-baked"
OUTPUT_LLM_GPTQ = "./SalesS2S-LLM-GPTQ-baked"
TEMP_LLM_DIR = "./temp_llm_extraction"

HF_MAIN_REPO = "kimballwightman/SalesS2S"
HF_LLM_REPO = "kimballwightman/SalesS2S-LLM-GPTQ"

# Calibration sentences for LLM quantization (sales domain)
CALIBRATION_TEXTS = [
    "Hi, I'm here to talk about our pest control service for your home.",
    "We offer free inspections and can help with ants, roaches, and rodents.",
    "Would you like me to explain our monthly protection plan?",
    "How long have you been dealing with this issue?",
    "Our service includes quarterly treatments and 24/7 emergency support.",
    "I understand your concerns about chemicals. We use eco-friendly options.",
    "The initial treatment takes about 45 minutes for an average home.",
    "We've been serving this neighborhood for over 15 years now.",
    "Many of your neighbors are already protected under our service plan.",
    "The warranty covers re-treatments if pests return between visits.",
]

print("🚀 SalesS2S Baking Process")
print("=" * 70)
print("This will create TWO quantized repos:")
print(f"  1. {HF_MAIN_REPO} - Complete model (WavLM + quantized LLM)")
print(f"  2. {HF_LLM_REPO} - Quantized LLM only (for runtime pattern)")
print("=" * 70)

# Load tokenizer (needed for calibration)
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_REPO, trust_remote_code=True)
print("✅ Tokenizer loaded")

# ============================================================
# STEP 1: Extract and quantize the LLM
# ============================================================
print("\n⚙️ STEP 1: Extracting and quantizing LLM to 8-bit...")
print("   Loading original model (this downloads ~15-20GB if not cached)...")

# Load original model to extract LLM
original_model = OmniSpeechModel.from_pretrained(
    ORIGINAL_REPO,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Keep on CPU during extraction
    trust_remote_code=True
)

print("✅ Original model loaded")
print(f"   Components: {type(original_model.audio_encoder_model).__name__} (audio) + "
      f"{type(original_model.llm_model).__name__} (LLM) + TTS")

# Extract and save LLM separately (needed for AutoGPTQ)
print(f"\n   Extracting LLM to {TEMP_LLM_DIR}...")
os.makedirs(TEMP_LLM_DIR, exist_ok=True)
llm_model = original_model.llm_model
llm_model.save_pretrained(TEMP_LLM_DIR)
tokenizer.save_pretrained(TEMP_LLM_DIR)

# Clear memory
del original_model
del llm_model
torch.cuda.empty_cache()

print("✅ LLM extracted")

# Prepare calibration dataset for quantization
print("\n   Preparing calibration dataset...")
examples = []
for text in CALIBRATION_TEXTS:
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    examples.append(tokens)

print(f"   Calibration dataset: {len(examples)} examples")

# Configure 8-bit quantization
print("\n   Configuring AutoGPTQ (8-bit, group_size=128)...")
quantize_config = BaseQuantizeConfig(
    bits=8,
    group_size=128,
    desc_act=False,
    sym=True,
    damp_percent=0.01
)

# Load LLM with quantization config
print("   Loading LLM into AutoGPTQ quantizer...")
llm_quantizer = AutoGPTQForCausalLM.from_pretrained(
    TEMP_LLM_DIR,
    quantize_config=quantize_config,
    device_map="auto"
)

# Run quantization (this is the slow step)
print("\n   🔥 Running quantization (this takes 10-20 minutes)...")
print("   GPU will be fully utilized during this step...")
llm_quantizer.quantize(examples)

print("\n✅ LLM quantized to 8-bit!")
print(f"   Memory reduction: ~14GB → ~7GB (~50% smaller)")

# Save quantized LLM separately
print(f"\n   Saving quantized LLM to {OUTPUT_LLM_GPTQ}...")
os.makedirs(OUTPUT_LLM_GPTQ, exist_ok=True)
llm_quantizer.save_quantized(OUTPUT_LLM_GPTQ)
tokenizer.save_pretrained(OUTPUT_LLM_GPTQ)

print("✅ Quantized LLM saved (for separate HF repo)")

# Clear GPU memory to free ~7GB (good practice even with A100's 40GB)
print("\n   🧹 Clearing GPU memory (quantized LLM already saved to disk)...")
del llm_quantizer
torch.cuda.empty_cache()
print("✅ GPU memory cleared (~7GB freed, now have ~13GB headroom)")

# ============================================================
# STEP 2: Build complete model with WavLM + quantized LLM
# ============================================================
print("\n📦 STEP 2: Building complete model with WavLM and quantized LLM...")

# Load original model again (fresh)
print("   Loading original model fresh...")
model = OmniSpeechModel.from_pretrained(
    ORIGINAL_REPO,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Replace audio encoder with pretrained WavLM
print("\n   🎤 Replacing Qwen2AudioEncoder with pretrained WavLM...")
from modeling_audio_encoder import WavLMEncoder

# Load Microsoft's pretrained WavLM model
pretrained_wavlm = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus",
    torch_dtype=torch.bfloat16
)

# Create WavLMEncoder wrapper with correct config
wavlm_config = model.config.audio_encoder_config
wavlm_config.model_type = "wavlm"
wavlm_config.hidden_size = 768
wavlm_config.d_model = 1280
wavlm_config.num_hidden_layers = 12
wavlm_config.num_attention_heads = 12
wavlm_config.intermediate_size = 3072

# Build WavLM encoder and assign pretrained weights
wavlm_encoder = WavLMEncoder(wavlm_config)
wavlm_encoder.wavlm = pretrained_wavlm.to(model.device)
wavlm_encoder = wavlm_encoder.to(model.device)

# Replace in composite model
model.audio_encoder_model = wavlm_encoder
model.config.audio_encoder_config = wavlm_config

print("✅ WavLM encoder baked in (pretrained from microsoft/wavlm-base-plus)")

# Replace LLM with quantized version
print("\n   🔄 Replacing full precision LLM with quantized version...")
quantized_llm = AutoGPTQForCausalLM.from_quantized(
    OUTPUT_LLM_GPTQ,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    use_triton=False
)

model.llm_model = quantized_llm

print("✅ Quantized LLM baked into composite model")
print(f"   Final model: WavLM (768-dim) + Quantized LLM (8-bit) + TTS")

# ============================================================
# STEP 3: Save complete quantized model
# ============================================================
print(f"\n💾 STEP 3: Saving complete model to {OUTPUT_MAIN}...")

os.makedirs(OUTPUT_MAIN, exist_ok=True)

try:
    # Save composite model with all quantized components
    model.save_pretrained(OUTPUT_MAIN, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(OUTPUT_MAIN)
    print("✅ Complete quantized model saved!")
    main_model_saved = True
except Exception as e:
    print(f"⚠️  Warning: Could not save composite model with quantized LLM")
    print(f"   Error: {e}")
    print("   The separate LLM-GPTQ repo will still work for runtime loading")
    main_model_saved = False

# ============================================================
# STEP 4: Upload to private HuggingFace repos
# ============================================================
print("\n☁️  STEP 4: Uploading to HuggingFace...")

api = HfApi()
token = os.environ.get("HF_TOKEN")

if not token:
    print("⚠️  No HF_TOKEN environment variable found")
    print("   Skipping upload to HuggingFace")
    print("   To upload, set: export HF_TOKEN=your_token")
    print(f"\n   Local files saved to:")
    print(f"     - {OUTPUT_MAIN}")
    print(f"     - {OUTPUT_LLM_GPTQ}")
else:
    # Upload quantized LLM (always works)
    print(f"\n   📤 Uploading to {HF_LLM_REPO}...")
    try:
        create_repo(HF_LLM_REPO, private=True, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=OUTPUT_LLM_GPTQ,
            repo_id=HF_LLM_REPO,
            repo_type="model",
            token=token
        )
        print(f"✅ {HF_LLM_REPO} uploaded!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

    # Upload main model (if it saved successfully)
    if main_model_saved:
        print(f"\n   📤 Uploading to {HF_MAIN_REPO}...")
        try:
            create_repo(HF_MAIN_REPO, private=True, exist_ok=True, token=token)
            api.upload_folder(
                folder_path=OUTPUT_MAIN,
                repo_id=HF_MAIN_REPO,
                repo_type="model",
                token=token
            )
            print(f"✅ {HF_MAIN_REPO} uploaded!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

# ============================================================
# Cleanup
# ============================================================
print("\n🧹 Cleaning up temporary files...")
if os.path.exists(TEMP_LLM_DIR):
    shutil.rmtree(TEMP_LLM_DIR)
print("✅ Cleanup complete")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("🎉 BAKING COMPLETE!")
print("=" * 70)

if token and main_model_saved:
    print("\n✅ Both private repos created and uploaded:")
    print(f"   📦 {HF_MAIN_REPO}")
    print("      - Pretrained WavLM audio encoder (768-dim)")
    print("      - 8-bit quantized LLM (~7GB instead of ~14GB)")
    print("      - Original TTS LM and adapters")
    print(f"   📦 {HF_LLM_REPO}")
    print("      - 8-bit quantized LLM standalone")
    print("      - For runtime loading pattern")
elif token:
    print(f"\n✅ Quantized LLM repo uploaded: {HF_LLM_REPO}")
    print("⚠️  Main repo upload skipped (composite save failed)")
else:
    print("\n✅ Local files created:")
    print(f"   - {OUTPUT_MAIN}")
    print(f"   - {OUTPUT_LLM_GPTQ}")
    print("\nRun with HF_TOKEN set to upload to HuggingFace")

print("\n📋 Next steps:")
print("   1. Update startup.py MODELS_CONFIG with your private repos")
print("   2. model_worker.py already handles loading quantized LLM")
print("   3. Rebuild Docker container")
print("   4. Deploy and test!")

print("\n" + "=" * 70)
