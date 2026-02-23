# GPTQ Persistent Quantization Guide

## Overview

This guide uses **AutoGPTQ** for persistent on-disk quantization of the Qwen LLM component.
Unlike BitsAndBytes (runtime-only), GPTQ saves quantized weights to disk for faster loading.

## Why GPTQ?

| Feature | BitsAndBytes | GPTQ |
|---------|-------------|-----|
| **Quantization Type** | Runtime (every load) | Persistent (once) |
| **Disk Size** | 21GB (full precision) | ~8-10GB (quantized LLM) |
| **Load Time** | 2-5 min (quantizes each time) | 30-60s (loads quantized) |
| **Memory Usage** | ~14-21GB VRAM | ~12-14GB VRAM |
| **Bit Width** | 4-bit or 8-bit | 8-bit (this guide) |
| **Best For** | Quick experimentation | Production use |

**For your use case (8-bit, persistent):** GPTQ is the right choice ✅

## Architecture

```
/models/
├── OpenS2S/                 # Original unified checkpoint (21GB)
│   ├── model-*.safetensors  # Contains: encoder + LLM + TTS + adapters
│   ├── audio/ (configs)
│   └── tts/ (tokenizers)
├── OpenS2S-llm-gptq/        # Quantized LLM ONLY (~8-10GB) ← NEW
├── wavlm-base-plus/         # Cached WavLM (~400MB) ← Auto-cached
└── glm-4-voice-decoder/     # Voice decoder
```

## Setup Instructions

### Step 1: Initial Setup (First Time)

```bash
cd ~/OpenS2S
git pull origin main

# Rebuild container with AutoGPTQ
docker stop opens2s-server
docker rm opens2s-server
./build_and_run.sh

# Wait for models to download
docker logs -f opens2s-server
# Wait for: "✅ All 2 models are ready!"
```

### Step 2: Run GPTQ Quantization (One-Time, ~10 minutes)

**IMPORTANT:** Stop the inference server first to free GPU memory:
```bash
# Stop the server to free GPU memory for quantization
docker stop opens2s-server
```

```bash
# Start container and run quantization immediately
docker start opens2s-server
docker exec -it opens2s-server python3 /app/quantize_llm_gptq.py
```

**What happens:**
1. Loads OmniSpeech (21GB unified checkpoint)
2. Extracts LLM component (~15GB worth of weights)
3. Quantizes LLM to 8-bit GPTQ (~8-10GB)
4. Saves to `/models/OpenS2S-llm-gptq/`
5. Deletes temp files

**Expected output:**
```
╔════════════════════════════════════════════════════════════════╗
║   GPTQ Quantization for OmniSpeech LLM (Persistent 8-bit)    ║
╚════════════════════════════════════════════════════════════════╝

📦 Step 1/3: Extracting LLM from OmniSpeech Checkpoint
✅ Standalone LLM checkpoint saved

🔧 Step 2/3: Quantizing LLM with AutoGPTQ (8-bit)
✅ LLM quantized to 8-bit
✅ Quantized LLM saved

📊 Step 3/3: Verifying Quantization
Original checkpoint (full): 21.00 GB
Quantized LLM only: 9.00 GB
Compression ratio: 2.33x
✅ Quantization successful!

🎉 SUCCESS! LLM Quantization Complete
```

### Step 3: Restart to Use Quantized Model

```bash
docker restart opens2s-server

# Verify it's using GPTQ
docker logs opens2s-server | grep -A 5 "GPTQ"
```

**Expected logs:**
```
🔄 Loading GPTQ-quantized LLM from /models/OpenS2S-llm-gptq...
✅ GPTQ-quantized LLM loaded (8-bit, ~8-10GB)
   Expected total VRAM: ~12-14GB
```

### Step 4: Verify Memory Usage

```bash
docker exec opens2s-server nvidia-smi
```

**Expected:**
- Before: ~14-21GB VRAM
- After: **~12-14GB VRAM** ✅

### Step 5: Save VM Snapshot

**Critical:** Save a VM snapshot now to preserve:
- ✅ `/models/OpenS2S/` (original, 21GB)
- ✅ `/models/OpenS2S-llm-gptq/` (quantized LLM, 8-10GB)
- ✅ `/models/wavlm-base-plus/` (cached WavLM, 400MB)
- ✅ `/models/glm-4-voice-decoder/` (decoder)

Future VMs from this snapshot will have **instant startup** with quantized models! ⚡

## How It Works

### Loading Flow:

```
1. Check: Does /models/OpenS2S-llm-gptq exist?
   ├─ YES: Load OmniSpeech base + quantized LLM (12-14GB) ✨
   └─ NO:  Load full OmniSpeech (14-21GB) + show tip

2. Check: Does /models/wavlm-base-plus exist?
   ├─ YES: Load from cache (instant)
   └─ NO:  Download and cache for next time

3. Assemble complete model:
   - Audio Encoder: WavLM (bf16, ~1GB)
   - LLM: GPTQ-quantized (8-bit, ~8-10GB) ← KEY SAVINGS
   - TTS: Qwen (bf16, ~2-3GB)
   - Adapters: (bf16, ~1GB)
   = Total: ~12-14GB
```

## Troubleshooting

### Q: Still using 14-21GB memory?

**Check logs:**
```bash
docker logs opens2s-server | grep -i "gptq\|quantized"
```

If you see:
```
💡 No GPTQ-quantized LLM found
```

GPTQ quantization didn't complete. Re-run Step 2.

### Q: AutoGPTQ import error?

**Error:** `ImportError: No module named 'auto_gptq'`

**Solution:**
```bash
# Rebuild container (AutoGPTQ should be in Dockerfile)
docker stop opens2s-server
docker rm opens2s-server
./build_and_run.sh
```

### Q: Quantization takes too long (>15 min)?

GPTQ quantization can take time depending on calibration data.
This is expected for first-time setup. Future loads will be fast!

### Q: Can I delete the original checkpoint after quantization?

**NO!** The original checkpoint still contains:
- Audio encoder weights
- TTS weights
- Adapter weights

Only the LLM is in the separate quantized checkpoint.

## Performance Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First load | 2-5 min | 10 min (one-time) | One-time cost |
| Subsequent loads | 2-5 min | **30-60s** | **4-5x faster** ✅ |
| GPU Memory | 14-21GB | **12-14GB** | **20-30% less** ✅ |
| Disk Space | 21GB | 29-31GB | +8-10GB (worth it!) |
| Container rebuild | Re-download (slow) | Instant (cached) | **10x faster** ✅ |

## Files Changed

- `quantize_llm_gptq.py` - New: GPTQ quantization script (8-bit)
- `model_worker.py` - Updated: Loads GPTQ-quantized LLM if available
- `Dockerfile` - Updated: Installs AutoGPTQ
- `build_and_run.sh` - Already updated: Mounts `/models` from host

## Notes

- GPTQ quantization is **persistent** (saves to disk)
- 8-bit quantization provides good quality with ~50% size reduction
- Quantized model works across container rebuilds
- WavLM is automatically cached after first download
- VM snapshots capture all cached/quantized models
- Original checkpoint is kept for reference
