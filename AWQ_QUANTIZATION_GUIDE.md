# AWQ Persistent Quantization Guide

## Overview

This guide uses **AutoAWQ** for persistent on-disk quantization of the Qwen LLM component.
Unlike BitsAndBytes (runtime-only), AWQ saves quantized weights to disk for faster loading.

## Why AWQ?

| Feature | BitsAndBytes | AWQ |
|---------|-------------|-----|
| **Quantization Type** | Runtime (every load) | Persistent (once) |
| **Disk Size** | 21GB (full precision) | ~4-5GB (quantized LLM) |
| **Load Time** | 2-5 min (quantizes each time) | 30-60s (loads quantized) |
| **Memory Usage** | ~14-21GB VRAM | ~8-10GB VRAM |
| **Best For** | Quick experimentation | Production use |

**For your use case (fast loading, persistent):** AWQ is the right choice ✅

## Architecture

```
/models/
├── OpenS2S/                 # Original unified checkpoint (21GB)
│   ├── model-*.safetensors  # Contains: encoder + LLM + TTS + adapters
│   ├── audio/ (configs)
│   └── tts/ (tokenizers)
├── OpenS2S-llm-awq/         # Quantized LLM ONLY (~4-5GB) ← NEW
├── wavlm-base-plus/         # Cached WavLM (~400MB) ← Auto-cached
└── glm-4-voice-decoder/     # Voice decoder
```

## Setup Instructions

### Step 1: Initial Setup (First Time)

```bash
cd ~/OpenS2S
git pull origin main

# Rebuild container with AutoAWQ
docker stop opens2s-server
docker rm opens2s-server
./build_and_run.sh

# Wait for models to download
docker logs -f opens2s-server
# Wait for: "✅ All 2 models are ready!"
```

### Step 2: Run AWQ Quantization (One-Time, ~10 minutes)

```bash
# This extracts LLM from unified checkpoint and quantizes it
docker exec -it opens2s-server python3 /app/quantize_llm_awq.py
```

**What happens:**
1. Loads OmniSpeech (21GB unified checkpoint)
2. Extracts LLM component (~15GB worth of weights)
3. Quantizes LLM to 4-bit AWQ (~4-5GB)
4. Saves to `/models/OpenS2S-llm-awq/`
5. Deletes temp files

**Expected output:**
```
╔════════════════════════════════════════════════════════════════╗
║   AWQ Quantization for OmniSpeech LLM (Persistent on Disk)    ║
╚════════════════════════════════════════════════════════════════╝

📦 Step 1/3: Extracting LLM from OmniSpeech Checkpoint
✅ Standalone LLM checkpoint saved

🔧 Step 2/3: Quantizing LLM with AutoAWQ
✅ LLM quantized to 4-bit
✅ Quantized LLM saved

📊 Step 3/3: Verifying Quantization
Original checkpoint (full): 21.00 GB
Quantized LLM only: 4.50 GB
Compression ratio: 4.67x
✅ Quantization successful!

🎉 SUCCESS! LLM Quantization Complete
```

### Step 3: Restart to Use Quantized Model

```bash
docker restart opens2s-server

# Verify it's using AWQ
docker logs opens2s-server | grep -A 5 "AWQ"
```

**Expected logs:**
```
🔄 Loading AWQ-quantized LLM from /models/OpenS2S-llm-awq...
✅ AWQ-quantized LLM loaded (4-bit, ~4-5GB)
   Expected total VRAM: ~8-10GB
```

### Step 4: Verify Memory Usage

```bash
docker exec opens2s-server nvidia-smi
```

**Expected:**
- Before: ~14-21GB VRAM
- After: **~8-10GB VRAM** ✅

### Step 5: Save VM Snapshot

**Critical:** Save a VM snapshot now to preserve:
- ✅ `/models/OpenS2S/` (original, 21GB)
- ✅ `/models/OpenS2S-llm-awq/` (quantized LLM, 4-5GB)
- ✅ `/models/wavlm-base-plus/` (cached WavLM, 400MB)
- ✅ `/models/glm-4-voice-decoder/` (decoder)

Future VMs from this snapshot will have **instant startup** with quantized models! ⚡

## How It Works

### Loading Flow:

```
1. Check: Does /models/OpenS2S-llm-awq exist?
   ├─ YES: Load OmniSpeech base + quantized LLM (8-10GB) ✨
   └─ NO:  Load full OmniSpeech (14-21GB) + show tip

2. Check: Does /models/wavlm-base-plus exist?
   ├─ YES: Load from cache (instant)
   └─ NO:  Download and cache for next time

3. Assemble complete model:
   - Audio Encoder: WavLM (bf16, ~1GB)
   - LLM: AWQ-quantized (4-bit, ~4-5GB) ← KEY SAVINGS
   - TTS: Qwen (bf16, ~2-3GB)
   - Adapters: (bf16, ~1GB)
   = Total: ~8-10GB
```

## Troubleshooting

### Q: Still using 14-21GB memory?

**Check logs:**
```bash
docker logs opens2s-server | grep -i "awq\|quantized"
```

If you see:
```
💡 No AWQ-quantized LLM found
```

AWQ quantization didn't complete. Re-run Step 2.

### Q: AutoAWQ import error?

**Error:** `ImportError: No module named 'awq'`

**Solution:**
```bash
# Rebuild container (AutoAWQ should be in Dockerfile)
docker stop opens2s-server
docker rm opens2s-server
./build_and_run.sh
```

### Q: Quantization takes too long (>15 min)?

AWQ quantization can be slow without calibration data. Current implementation uses simple quantization.
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
| GPU Memory | 14-21GB | **8-10GB** | **40-50% less** ✅ |
| Disk Space | 21GB | 25GB | +4GB (worth it!) |
| Container rebuild | Re-download (slow) | Instant (cached) | **10x faster** ✅ |

## Files Changed

- `quantize_llm_awq.py` - New: AWQ quantization script
- `model_worker.py` - Updated: Loads AWQ-quantized LLM if available
- `Dockerfile` - Updated: Installs AutoAWQ
- `build_and_run.sh` - Already updated: Mounts `/models` from host

## Notes

- AWQ quantization is **persistent** (saves to disk)
- Quantized model works across container rebuilds
- WavLM is automatically cached after first download
- VM snapshots capture all cached/quantized models
- Original checkpoint is kept for reference
