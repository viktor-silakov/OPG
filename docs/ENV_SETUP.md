# Environment Variables Setup

This document describes all environment variables used by the Worker-Based Audio Generation System.

## 📋 Overview

The system uses environment variables for configuration to provide flexibility across different deployment environments. All variables have sensible defaults and can be set via `.env` file or system environment.

## 🔧 Core Configuration

### CONCURRENT_REQUESTS
**Description:** Number of concurrent audio worker processes  
**Type:** Integer  
**Default:** `2`  
**Range:** `1-8` (limited by available memory)  
**Example:**
```bash
export CONCURRENT_REQUESTS=3
```

**Recommendations:**
- **Development:** `1-2` (easier debugging)
- **Production:** `3-4` (optimal performance)
- **Low memory:** `1` (minimal resource usage)

### AUDIO_DEVICE
**Description:** Device for TTS processing  
**Type:** String  
**Default:** `mps`  
**Options:** `mps`, `cuda`, `cpu`  
**Example:**
```bash
export AUDIO_DEVICE=mps
```

**Device Selection:**
- **`mps`:** Apple Silicon GPUs (M1/M2/M3)
- **`cuda`:** NVIDIA GPUs
- **`cpu`:** CPU processing (slowest but most compatible)

### MODEL_VERSION
**Description:** Fish Speech model version  
**Type:** String  
**Default:** `1.5`  
**Options:** `1.4`, `1.5`, `1.6`  
**Example:**
```bash
export MODEL_VERSION=1.5
```

**Version Details:**
- **1.4:** Stable, older version
- **1.5:** Recommended, balanced performance/quality
- **1.6:** Latest, experimental features

### MODEL_PATH
**Description:** Path to custom fine-tuned model (overrides MODEL_VERSION)  
**Type:** String  
**Default:** `undefined` (uses official model)  
**Example:**
```bash
export MODEL_PATH=/path/to/custom/model
```

**Requirements:**
- Must contain `model.pth` and `config.json`
- Must be compatible with Fish Speech 1.5+

## 🚀 Performance Optimization

### SEMANTIC_TOKEN_CACHE
**Description:** Enable/disable semantic tokens caching  
**Type:** Boolean  
**Default:** `true`  
**Options:** `true`, `false`  
**Example:**
```bash
export SEMANTIC_TOKEN_CACHE=true
```

**Impact:**
- **`true`:** 2-3x faster for similar text, uses disk space
- **`false`:** Consistent performance, no disk cache

**When to disable:**
- Testing/development (predictable results)
- Limited disk space
- Debugging generation issues

### PyTorch Memory Management

#### PYTORCH_MPS_HIGH_WATERMARK_RATIO
**Description:** MPS memory allocation ratio  
**Type:** Float  
**Default:** `0.0`  
**Range:** `0.0-1.0`  
**Example:**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### PYTORCH_MPS_ALLOCATOR_POLICY
**Description:** MPS memory allocation policy  
**Type:** String  
**Default:** `garbage_collection`  
**Options:** `garbage_collection`, `native`  
**Example:**
```bash
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
```

### Threading Configuration

#### OMP_NUM_THREADS
**Description:** OpenMP thread count  
**Type:** Integer  
**Default:** `1`  
**Example:**
```bash
export OMP_NUM_THREADS=1
```

#### MKL_NUM_THREADS
**Description:** Intel MKL thread count  
**Type:** Integer  
**Default:** `1`  
**Example:**
```bash
export MKL_NUM_THREADS=1
```

#### OPENBLAS_NUM_THREADS
**Description:** OpenBLAS thread count  
**Type:** Integer  
**Default:** `1`  
**Example:**
```bash
export OPENBLAS_NUM_THREADS=1
```

**Note:** Set to `1` to prevent conflicts between multiple workers.

## 📁 Configuration Files

### .env File Setup
Create a `.env` file in the project root:

```bash
# Copy example configuration
cp backend/config/worker.config.example .env

# Edit values as needed
nano .env
```

### Example .env File
```bash
# Core configuration
CONCURRENT_REQUESTS=3
AUDIO_DEVICE=mps
MODEL_VERSION=1.5

# Performance optimization
SEMANTIC_TOKEN_CACHE=true

# Memory management
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

# Threading
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

## 🎯 Preset Configurations

### Development/Testing
```bash
export CONCURRENT_REQUESTS=1
export AUDIO_DEVICE=cpu
export SEMANTIC_TOKEN_CACHE=false
export MODEL_VERSION=1.5
```

### Production (High Performance)
```bash
export CONCURRENT_REQUESTS=4
export AUDIO_DEVICE=mps
export SEMANTIC_TOKEN_CACHE=true
export MODEL_VERSION=1.5
```

### Memory Constrained
```bash
export CONCURRENT_REQUESTS=1
export AUDIO_DEVICE=mps
export SEMANTIC_TOKEN_CACHE=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Batch Processing
```bash
export CONCURRENT_REQUESTS=3
export AUDIO_DEVICE=mps
export SEMANTIC_TOKEN_CACHE=true
export MODEL_VERSION=1.5
```

## 🔍 Validation and Debugging

### Check Current Configuration
```bash
# Show all environment variables
env | grep -E "(CONCURRENT_REQUESTS|AUDIO_DEVICE|MODEL_VERSION|SEMANTIC_TOKEN_CACHE)"

# Test device availability
cd fs-python
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Verify Model Loading
```bash
cd fs-python
python -c "from cli_tts import setup_fish_speech; print('Model setup:', setup_fish_speech('1.5') is not None)"
```

### Cache Management
```bash
# Check cache size
du -sh fs-python/cache/semantic_tokens/ 2>/dev/null || echo "No cache"

# Clear cache
rm -rf fs-python/cache/semantic_tokens/*

# Disable cache temporarily
SEMANTIC_TOKEN_CACHE=false npm run test:worker
```

## 📊 Performance Impact

| Variable | Impact | Memory | Speed | Disk |
|----------|--------|--------|-------|------|
| CONCURRENT_REQUESTS=1 | Low resource usage | ✅ Low | ❌ Slow | ✅ Low |
| CONCURRENT_REQUESTS=4 | High performance | ❌ High | ✅ Fast | ✅ Low |
| AUDIO_DEVICE=cpu | Compatible | ✅ Low | ❌ Slow | ✅ Low |
| AUDIO_DEVICE=mps | Optimal for Apple | ❌ High | ✅ Fast | ✅ Low |
| SEMANTIC_TOKEN_CACHE=true | Fast repeat | ❌ Medium | ✅ Fast | ❌ High |
| SEMANTIC_TOKEN_CACHE=false | Predictable | ✅ Low | ❌ Slow | ✅ Low |

## 🚨 Troubleshooting

### Workers Won't Start
```bash
# Check device availability
export AUDIO_DEVICE=cpu
npm run test:worker

# Reduce worker count
export CONCURRENT_REQUESTS=1
npm run test:worker
```

### Out of Memory
```bash
# Reduce workers
export CONCURRENT_REQUESTS=1

# Use CPU
export AUDIO_DEVICE=cpu

# Disable cache
export SEMANTIC_TOKEN_CACHE=false

# Optimize MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Slow Performance
```bash
# Enable cache
export SEMANTIC_TOKEN_CACHE=true

# Use GPU
export AUDIO_DEVICE=mps  # or cuda

# Increase workers (if memory allows)
export CONCURRENT_REQUESTS=3
```

### Cache Issues
```bash
# Clear cache
rm -rf fs-python/cache/semantic_tokens/*

# Disable cache temporarily
export SEMANTIC_TOKEN_CACHE=false

# Check cache size
du -sh fs-python/cache/semantic_tokens/
```

## 📝 Notes

- Environment variables are read at startup - restart required for changes
- `.env` file variables override system environment
- Invalid values fall back to defaults with warnings
- Cache location: `fs-python/cache/semantic_tokens/`
- Model cache location: `~/.cache/huggingface/hub/` 