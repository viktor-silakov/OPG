# Fish Speech Fine-tuning Guide

Complete guide for fine-tuning Fish Speech model to create custom voices.

## Contents

1. [Process Overview](#process-overview)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Model Fine-tuning](#model-fine-tuning)
5. [Resume Training](#resume-training)
6. [Using the Model](#using-the-model)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Process Overview

Fish Speech fine-tuning consists of 5 main stages:

1. **Data Preparation** - converting audio to the required format
2. **Semantic Token Extraction** - encoding audio into tokens
3. **Protobuf Dataset Creation** - packaging data for training
4. **LoRA Fine-tuning** - adapting the model to your data
5. **Weight Merging** - creating the final model

## Requirements

### System Requirements

- **Python 3.10+**
- **GPU**: minimum 8GB VRAM (for fine-tuning), 4GB (for inference)
- **RAM**: minimum 16GB
- **Disk Space**: 10-20GB (depends on dataset size)
- **Time**: 1-4 hours (depends on dataset size and GPU)

### Dependencies

```bash
# Main dependencies for data preparation
poetry run pip install librosa soundfile whisper tqdm yt-dlp

# For downloading videos from YouTube (optional)
pip install yt-dlp
```

### Data Quality

- **Minimum**: 10-30 minutes of high-quality audio
- **Recommended**: 30-60 minutes of diverse content
- **Format**: 44.1kHz, mono, WAV/MP3/FLAC
- **Clarity**: minimal background noise, clear speech
- **Text**: accurate transcription for each audio file

## Data Preparation

### Automatic Preparation

```bash
cd fs-python

# Processing local audio folder
poetry run python prepare_dataset.py \
  --input samples \
  --output training_data/note_lm \
  --normalize \
  --split-long \
  --auto-transcribe

# Download and process from YouTube (automatic 10s segmentation)
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --segment-duration 10

# Custom segment duration (e.g., 15 seconds)
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --segment-duration 15

# Process only first 20 minutes of video
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --max-duration 20

# Combination: first 30 minutes + 12-second segments
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --segment-duration 12 \
  --max-duration 30
```

### Manual Preparation

If you already have prepared data, organize it like this:

```
training_data/
└── my_voice/
    ├── SPEAKER_NAME/
    │   ├── audio_001.wav
    │   ├── audio_001.lab
    │   ├── audio_002.wav
    │   ├── audio_002.lab
    │   └── ...
    └── dataset_summary.json
```

Where:
- `.wav` files contain audio segments (preferably 10-30 seconds)
- `.lab` files contain accurate audio transcription
- Filenames must match (except extension)

## Model Fine-tuning

### Full Automatic Pipeline

```bash
# Run complete training process
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_voice \
  --model-version 1.5 \
  --full-pipeline \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 1e-4
```

### Step-by-Step Process

```bash
# Step 1: Data preparation (if not done)
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir raw_audio/ \
  --prepare-data

# Step 2: Extract semantic tokens
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_custom_voice \
  --extract-tokens \
  --batch-size-extract 16

# Step 3: Train model
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_custom_voice \
  --train \
  --max-steps 200 \
  --batch-size 2 \
  --learning-rate 5e-5

# Step 4: Merge weights
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --merge-weights \
  --data-dir training_data/my_custom_voice

poetry run python finetune_tts.py \
  --project note_lm_step2 \
  --merge-weights \
  --data-dir training_data/note_lm
```

### Safe Training Parameters

⚠️ **Important**: Incorrect parameters can lead to overfitting or model instability!

| Parameter | "Safe" Value | Comment |
|-----------|--------------|---------|
| **VRAM** | ≥ 8 GB for fine-tune | Verified in official documentation |
| **batch_size** | 2-4 (with 8 GB) | With gradient_accumulation_steps easily increase "logical" batch |
| **learning_rate** | 1e-5 – 5e-5 | Higher LR causes noise / "collapse" after 300 steps |
| **max_steps** | 100-300 | Enough for model to "catch" intonation; further — overfitting risk |
| **LoRA rank / α** | r_8_alpha_16 | Preset configuration in guide |

### ⚠️ Important Fine-tuning Limitations

**By default, fine-tune trains pronunciation, but not timbre.**

- **For timbre need**: more steps (≈ 500-1000) + diverse prompts
- **Risk**: without this, voice may "drift" - become unstable or unnatural
- **Recommendation**: start with 100-300 steps to check quality, then increase if needed

**Signs of Overfitting:**
- Voice becomes robotic
- Artifacts and noise appear
- Model loses speech naturalness
- Loss stops decreasing or starts growing

### Training Strategies for Different Goals

#### 🎯 For Learning Pronunciation (fast and safe)
```bash
poetry run python finetune_tts.py \
  --project quick_pronunciation \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 100 \
  --batch-size 2 \
  --learning-rate 2e-5
```
**Result**: Model learns basic pronunciation in 15-30 minutes

#### 🎤 For Capturing Voice Timbre (slow, requires caution)
```bash
# First stage: basic training
poetry run python finetune_tts.py \
  --project voice_timbre_stage1 \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 300 \
  --batch-size 2 \
  --learning-rate 2e-5

# Second stage: fine-tune timbre
poetry run python finetune_tts.py \
  --project voice_timbre_stage2 \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 500 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --checkpoint-path results/voice_timbre_stage1/checkpoints/
```
**Result**: More accurate timbre reproduction, but overfitting risk

## Resume Training

### 🚀 Resume Training Features

Now you can:
- ✅ **Continue training** from any checkpoint
- ✅ **Automatically find** the latest checkpoint
- ✅ **View a list** of all available checkpoints
- ✅ **Flexibly manage** the training process

### 📂 Viewing Available Checkpoints

View all available checkpoints for a project:

```bash
poetry run python finetune_tts.py \
  --project note_lm_step2 \
  --data-dir training_data/note_lm \
  --list-checkpoints
```

**Example output:**
```
📂 Available checkpoints for project 'note_lm_step2':
📄 step_000000100.ckpt (71.0 MB, 2025-06-08 07:30:28)
📄 step_000000200.ckpt (71.0 MB, 2025-06-08 08:28:52)
🎯 Latest: step_000000200.ckpt
```

### 🔄 Resuming Training

#### 1. Automatically from Latest Checkpoint

```bash
poetry run python finetune_tts.py \
  --project note_lm_step3 \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 300 \
  --batch-size 8 \
  --learning-rate 1.2e-5 \
  --resume-latest
```

#### 2. From Specific Checkpoint

```bash
poetry run python finetune_tts.py \
  --project note_lm_step3 \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 300 \
  --batch-size 8 \
  --learning-rate 1.2e-5 \
  --resume-from-checkpoint fish-speech/results/note_lm_step2/checkpoints/step_000000100.ckpt
```

### 💡 Recommended Training Scenarios

#### Gradual Training with Increasing Learning Rate

```bash
# Stage 1: Careful start
poetry run python finetune_tts.py \
  --project note_lm_stage1 \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 200 \
  --batch-size 4 \
  --learning-rate 5e-6

# Stage 2: Increase intensity
poetry run python finetune_tts.py \
  --project note_lm_stage2 \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 300 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --resume-latest

# Stage 3: Final fine-tuning
poetry run python finetune_tts.py \
  --project note_lm_final \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 200 \
  --batch-size 8 \
  --learning-rate 8e-6 \
  --resume-latest
```

#### Experimenting with Hyperparameters

```bash
# Check available checkpoints
poetry run python finetune_tts.py \
  --project note_lm_base \
  --data-dir training_data/note_lm \
  --list-checkpoints

# Experiment with different learning rates
poetry run python finetune_tts.py \
  --project note_lm_exp1 \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 150 \
  --batch-size 6 \
  --learning-rate 1.5e-5 \
  --resume-from-checkpoint fish-speech/results/note_lm_base/checkpoints/step_000000100.ckpt

# Parallel experiment with other parameters
poetry run python finetune_tts.py \
  --project note_lm_exp2 \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 150 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --resume-from-checkpoint fish-speech/results/note_lm_base/checkpoints/step_000000100.ckpt
```

### ⚠️ Important Notes

#### 1. Project Names
- **Use different names** when continuing training
- Example: `note_lm_step1` → `note_lm_step2` → `note_lm_final`

#### 2. Learning Rate
- When resuming training, you can often **increase the learning rate**
- Start with conservative values and gradually increase

#### 3. Validation
- **Test the model** after each training stage
- Save checkpoints with good results

### 🎯 Practical Example for Your Data

Based on your previous training log:

```bash
# 1. Continue with higher learning rate
poetry run python finetune_tts.py \
  --project note_lm_improved \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 250 \
  --batch-size 8 \
  --learning-rate 1.2e-5 \
  --resume-latest

# 2. If all goes well - another stage
poetry run python finetune_tts.py \
  --project note_lm_final \
  --data-dir training_data/note_lm \
  --train \
  --max-steps 200 \
  --batch-size 8 \
  --learning-rate 8e-6 \
  --resume-latest
```

### 🔧 Troubleshooting Resume Training

#### Checkpoint Not Found
```bash
# Check available checkpoints
poetry run python finetune_tts.py \
  --project your_project \
  --data-dir training_data/note_lm \
  --list-checkpoints
```

#### Parameter Errors
- Cannot use `--resume-latest` and `--resume-from-checkpoint` simultaneously
- Ensure the checkpoint path exists

#### Hallucinations After Resume Training
- Reduce learning rate by half
- Use an earlier checkpoint
- Reduce the number of training steps

### 📈 Progress Monitoring

Monitor these metrics:
- **Loss should decrease** (but slowly)
- **Accuracy should increase** (but gradually)
- **Validation should be close to train** (no overfitting)

Good progress: loss decreases by 0.1-0.3 over 200-300 steps.

## Using the Model

### After Fine-tuning Completion

The trained model will be saved in:
```
checkpoints/my_custom_voice-merged/
├── model.pth              # Main model (~1.2GB)
├── tokenizer.tiktoken     # Tokenizer (~1.6MB)
├── config.json            # Model configuration
└── special_tokens.json    # Special tokens
```

### Integration with CLI TTS

Update `cli_tts.py` to use your model:

```python
# In setup_fish_speech() function add path to your model
custom_model_path = "checkpoints/my_custom_voice-merged"
if Path(custom_model_path).exists():
    print(f"🎤 Using trained model: {custom_model_path}")
    return fish_speech_dir, Path(custom_model_path)
```

**Alternatively**, specify the path directly when calling TTS:

```bash
# Use full path to your model
poetry run python cli_tts.py "Testing our trained model" \
  --model-path checkpoints/test_limited-merged \
  -o output/test_finetuned.wav \
  --play
```

### Testing the Model

```bash
# Test with custom model
poetry run python cli_tts.py "Testing our trained model" \
  --model-path checkpoints/my_custom_voice-merged \
  -o output/test_finetuned.wav \
  --play
```

## Examples

### Example 1: Training on Podcast Recordings

```bash
# 1. Download podcast from YouTube
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=PODCAST_ID" \
  --output training_data/podcast_host \
  --speaker "PodcastHost" \
  --youtube \
  --auto-transcribe \
  --whisper-model large \
  --split-long

# 2. Run training
poetry run python finetune_tts.py \
  --project podcast_voice \
  --data-dir training_data/podcast_host \
  --full-pipeline \
  --max-steps 1500 \
  --batch-size 4
```

### Example 2: Training on Audiobook

```bash
# 1. Prepare audiobook chapters
poetry run python prepare_dataset.py \
  --input samples/ \
  --output training_data/note_lm \
  --normalize \
  --split-long \
  --auto-transcribe \
  --whisper-model base

# 2. Train with lower learning rate (for more stable voice)
poetry run python finetune_tts.py \
  --project audiobook_narrator \
  --data-dir training_data/narrator \
  --full-pipeline \
  --max-steps 200 \
  --batch-size 2 \
  --learning-rate 2e-5
```

### Example 3: Training on Your Own Voice

```bash
# 1. Record your voice (recommended 30-45 minutes of diverse content)
# Save as: my_voice/recordings/session_01.wav, session_02.wav, etc.
# Create text files: session_01.txt, session_02.txt

# 2. Prepare data
poetry run python prepare_dataset.py \
  --input my_voice/recordings \
  --output training_data/my_voice \
  --speaker "MyVoice" \
  --normalize \
  --split-long

# 3. Train with conservative settings
poetry run python finetune_tts.py \
  --project my_personal_voice \
  --data-dir training_data/my_voice \
  --full-pipeline \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --device mps  # for Apple Silicon
```

## Training Monitoring

### Training Logs

Logs are saved in:
```
fish-speech/results/my_custom_voice/
├── checkpoints/
├── logs/
└── training_configs/
```

### Key Metrics

Monitor:
- **Loss**: should gradually decrease
- **Learning Rate**: automatically adapts
- **GPU Memory**: should not exceed available

### Stopping Training

Training can be stopped at any time with `Ctrl+C`. Checkpoints are saved every 100 steps.

## Troubleshooting

### Common Problems

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
--batch-size 1

# Or use gradient accumulation
--gradient-accumulation-steps 4
```

**2. Model Not Converging**
```bash
# Reduce learning rate
--learning-rate 5e-5

# Increase number of steps
--max-steps 2000
```

**3. Poor Quality Synthesis**
```bash
# Check data quality
poetry run python prepare_dataset.py --input training_data/ --output check_data/ --auto-transcribe

# Try different checkpoint
--checkpoint-step 500  # instead of latest
```

**4. Slow Training**
```bash
# Use smaller dataset for testing
--max-steps 500

# Check GPU usage
nvidia-smi  # for NVIDIA
```

### Data Quality Check

```bash
# Analyze prepared dataset
poetry run python -c "
import json
with open('training_data/my_voice/dataset_summary.json') as f:
    summary = json.load(f)
    print(f'Files: {summary["total_files"]}')
    print(f'Duration: {summary["total_duration_minutes"]} min')
    print(f'Speakers: {summary["speakers"]}')
"
```

### Performance Optimization

**For Apple Silicon (MPS):**
```bash
--device mps
--batch-size 2  # MPS may be less stable with large batches
```

**For NVIDIA GPU:**
```bash
--device cuda
--batch-size 4  # or more if memory allows
```

**For CPU (slow):**
```bash
--device cpu
--batch-size 1
--max-steps 100  # for testing
```

### Specific Issues

#### Data Type Error in LoRA Layers

**Symptoms:**
```
RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float
```

**Cause:** 
Pretrained Fish Speech model loads with BFloat16 precision, while LoRA adapters initialize in Float32. This leads to data type incompatibility in matrix operations.

**Solutions:**

1. **Automatic (recommended):** Script automatically uses FP32 for all operations
   ```bash
   # Script automatically sets correct precision
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train
   ```

2. **If error persists:** Force device and precision
   ```bash
   # Explicit precision setup
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train --device cpu --batch-size 1
   ```

**Technical Details:**
- Base Fish Speech model saved with BFloat16 precision
- LoRA layers default to Float32 creation
- Script forces FP32 for all components (`trainer.precision=32` + `model.torch_dtype=float32`)
- This may slow training but ensures stability

#### Tensor Errors on Apple Silicon (MPS)

**Symptoms:**
```
RuntimeError: Expected scalar_type == ScalarType::Float || inputTensor.scalar_type() == ScalarType::Int || scalar_type == ScalarType::Bool to be true, but got false.
```

**Cause:** 
MPS (Metal Performance Shaders) on Apple Silicon has compatibility limitations with some PyTorch operations, especially with mixed precision training and certain tensor types.

**Solutions:**

1. **Automatic (recommended):** Script automatically switches to CPU when Apple Silicon detected
   ```bash
   # CPU used automatically on Apple Silicon
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train
   ```

2. **Force MPS usage (experimental):**
   ```bash
   # Try MPS with full precision
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train --force-mps
   ```

3. **Explicit CPU usage:**
   ```bash
   # Force CPU usage
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train --device cpu
   ```

**Performance on Apple Silicon:**
- **CPU:** Stable, supports all operations, slower than GPU
- **MPS:** Faster than CPU, but may be unstable with some models
- Recommended: start with CPU, try `--force-mps` if needed

#### Slow Training

**Symptoms:** Training progresses very slowly

**Solutions:**
1. Reduce batch size: `--batch-size 1`
2. Reduce number of steps: `--max-steps 500`
3. Use smaller LoRA rank: `--lora-config r_4_alpha_8`
4. If NVIDIA GPU available, use `--device cuda`

#### Memory Issues

**Symptoms:** 
```
RuntimeError: [enforce fail at alloc_cpu.cpp] data.
OutOfMemoryError: Unable to allocate array
```

**Solutions:**
1. Reduce batch size to 1: `--batch-size 1`
2. Use CPU instead of GPU: `--device cpu`
3. Close other applications
4. Reduce number of workers: `--num-workers-extract 1`

## Advanced Techniques

### Voice Mixing

You can train model on multiple voices:

```bash
# Prepare data for each speaker separately
poetry run python prepare_dataset.py --input speaker1_data/ --output training_data/multi_voice --speaker Speaker1
poetry run python prepare_dataset.py --input speaker2_data/ --output training_data/multi_voice --speaker Speaker2

# Train on combined dataset
poetry run python finetune_tts.py \
  --project multi_voice_model \
  --data-dir training_data/multi_voice \
  --full-pipeline
```

### Incremental Training

```bash
# Continue training existing model
poetry run python finetune_tts.py \
  --project my_voice_v2 \
  --data-dir new_training_data/ \
  --train \
  --pretrained-ckpt-path checkpoints/my_custom_voice-merged/
```

### Trained Model Verification

After training completion, verify all files are created correctly:

```bash
# Check model file sizes
ls -lh checkpoints/my_custom_voice-merged/

# Should show approximately:
# model.pth              ~1.2GB   # Main model
# tokenizer.tiktoken     ~1.6MB   # Tokenizer
# config.json            ~1KB     # Configuration
# special_tokens.json    ~30KB    # Special tokens
```

**Expected Sizes:**
- `model.pth`: 1.0-1.3 GB (depends on base model size)
- `tokenizer.tiktoken`: 1-2 MB (token vocabulary)
- `config.json`: less than 1 KB (model parameters)
- `special_tokens.json`: 20-40 KB (special tokens)

If any file is missing or has unexpected size, check training logs.

### LoRA Experiments

```bash
# More aggressive training
--lora-config r_16_alpha_32

# More conservative training  
--lora-config r_4_alpha_8
```

## Conclusion

Fish Speech fine-tuning allows creating high-quality custom voices. Key success factors:

1. **Quality Data**: clean recordings with accurate transcription
2. **Sufficient Volume**: minimum 10-30 minutes of audio
3. **Correct Parameters**: start with conservative settings
4. **Patience**: good results require time

Happy fine-tuning! 🎤✨