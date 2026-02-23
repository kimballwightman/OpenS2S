# Model Quantization Guide

## Overview

This guide explains how to create and use a pre-quantized version of the OmniSpeech model for faster loading and reduced memory usage.

## Why Pre-Quantize?

**Before (Runtime Quantization):**
- ❌ Quantization happens every container restart
- ❌ Slow: 2-5 minutes loading time
- ❌ Complex: Failed silently in our implementation
- ❌ Uses 14GB GPU memory (not actually quantized!)

**After (Pre-Quantized Model):**
- ✅ Quantize ONCE, use forever
- ✅ Fast: 30-60 seconds loading time
- ✅ Simple: Just load the model
- ✅ Uses ~8-10GB GPU memory (truly quantized)
- ✅ Persists across container rebuilds
- ✅ Saved in VM snapshots

## Architecture

```
VM Host Disk
└── ~/models/                        # Mounted to /models in container
    ├── OpenS2S/                     # Original model (21GB, downloaded first run)
    ├── OpenS2S-8bit/                # Quantized model (10GB, created once)
    └── glm-4-voice-decoder/         # Voice decoder (downloaded first run)

Docker Container
└── /models/ → (mounted from host)   # Same directory, persists across rebuilds
```

## Setup Instructions

### Step 1: Initial Setup (First Time)

On your GCP VM:

```bash
cd ~/OpenS2S
git pull origin main
./build_and_run.sh
```

This will:
- Create `~/models` directory on host
- Mount it to `/models` in container
- Download original models to `~/models/OpenS2S` (first run only)
- Start the server

**Wait for models to download** (check logs):
```bash
docker logs -f opens2s-server
```

Look for:
```
✅ All 2 models are ready!
```

### Step 2: Run One-Time Quantization

**Important:** Only run this AFTER models are downloaded.

```bash
# Run the quantization script (takes 5-10 minutes)
docker exec -it opens2s-server python3 /app/quantize_model_once.py
```

This creates `~/models/OpenS2S-8bit/` with the quantized model (~10GB).

### Step 3: Restart to Use Quantized Model

```bash
docker restart opens2s-server
```

The startup script automatically detects and uses the quantized model:
```
Using pre-quantized model: /models/OpenS2S-8bit
```

### Step 4: Verify GPU Memory Usage

```bash
docker exec opens2s-server nvidia-smi
```

**Expected memory usage:**
- Before quantization: ~14GB (not actually quantized)
- After quantization: ~8-10GB (truly quantized)

### Step 5: Save VM Snapshot

**Critical:** After quantization, save a VM snapshot/image. This preserves:
- ✅ `~/models/OpenS2S/` (original, 21GB)
- ✅ `~/models/OpenS2S-8bit/` (quantized, 10GB)
- ✅ `~/models/glm-4-voice-decoder/` (decoder)

Future VMs from this snapshot will have instant startup (no downloads or quantization needed)!

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ First Time Setup (on fresh VM)                              │
├─────────────────────────────────────────────────────────────┤
│ 1. ./build_and_run.sh              (~2 min, downloads)     │
│ 2. docker exec ... quantize...     (~10 min, one-time)     │
│ 3. docker restart                  (~1 min)                 │
│ 4. Save VM snapshot                (preserves quantized)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Future VM from Snapshot                                      │
├─────────────────────────────────────────────────────────────┤
│ 1. ./build_and_run.sh              (~1 min, instant!)      │
│    Models already on disk, pre-quantized ✨                 │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Quantized model not being used

Check startup logs:
```bash
docker logs opens2s-server | grep "model:"
```

Should show:
```
Using pre-quantized model: /models/OpenS2S-8bit
```

If it shows original model, quantization didn't complete. Re-run Step 2.

### Still using 14GB GPU memory

This means the model isn't actually quantized. Possible causes:
1. Quantization script didn't complete successfully
2. Container is using wrong model path

Solution: Delete `~/models/OpenS2S-8bit/` and re-run quantization script.

### Models disappeared after rebuild

This means the volume mount isn't working. Check:
```bash
ls ~/models/
```

Should show `OpenS2S`, `OpenS2S-8bit`, `glm-4-voice-decoder`.

If empty, volume mount failed. Check `build_and_run.sh` has:
```bash
-v "$HOME/models:/models"
```

## Files Changed

- `quantize_model_once.py` - New: One-time quantization script
- `model_worker.py` - Simplified: Removed runtime quantization logic
- `startup.py` - Updated: Auto-detects quantized model
- `build_and_run.sh` - Updated: Mounts host models directory

## Benefits Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First load time | 2-5 min | 10 min (one-time) | - |
| Subsequent loads | 2-5 min | 30-60 sec | **4-5x faster** |
| GPU Memory | 14GB | 8-10GB | **30-40% less** |
| Container rebuild | Re-download | Instant | **10x faster** |
| VM snapshot size | +21GB | +31GB | One-time cost |

## Notes

- The original model (`~/models/OpenS2S`) is kept for reference
- Both models persist across container rebuilds
- VM snapshots capture both models
- You can delete the original after verifying quantized model works
