# Fish Speech TTS with MPS

Text-to-Speech implementation using Fish Speech model with Metal Performance Shaders (MPS) acceleration for Apple Silicon Macs.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.9+** 
- **Poetry** for dependency management
- **Git** for cloning repositories
- At least **2GB** of free disk space (model stored in Hugging Face cache)
- Stable internet connection for model download

## Installation

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone and setup the project**:
   ```bash
   git clone <your-repo-url>
   cd fs-python
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Install Fish Speech in editable mode**:
   ```bash
   poetry run pip install -e ./fish-speech
   ```

5. **Install fine-tuning dependencies** (optional, for training custom voices):
   ```bash
   poetry run pip install librosa soundfile whisper tqdm yt-dlp
   ```

## Table of Contents

- [Usage](#usage)
- [Fine-tuning Custom Voices](#fine-tuning-custom-voices)
- [Voice Cloning](#voice-cloning)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Usage

### Command Line Interface (CLI)

The simplest way to use Fish Speech TTS:

```bash
# Basic usage
poetry run python cli_tts.py "Your text to synthesize"

# Specify output file (saved to output/ directory by default)
poetry run python cli_tts.py "Hello world" -o output/hello.wav

# Auto-play after synthesis
poetry run python cli_tts.py "Hello world" --play

# Use different device (CPU instead of MPS)
poetry run python cli_tts.py "Hello world" --device cpu
```

**CLI Options:**
- `text` - Text to synthesize (required)
- `-o, --output` - Output WAV file (default: output/output.wav)
- `--device` - Computing device: mps, cpu, cuda (default: mps)
- `--play` - Auto-play audio after synthesis

### 🔥 Flash-Optimized CLI (Recommended)

For enhanced performance and memory efficiency, use the Flash-Optimized version:

```bash
# Drop-in replacement for cli_tts.py with Flash Attention optimizations
poetry run python flash_optimized_cli.py "Your text to synthesize"

# All cli_tts.py arguments work identically
poetry run python flash_optimized_cli.py "Hello world" -o output/hello.wav --play

# Additional Flash Attention features
poetry run python flash_optimized_cli.py "Long text generation" --monitor
poetry run python flash_optimized_cli.py --benchmark
```

**🚀 Key Benefits:**
- **30-50% less memory usage** through Flash Attention O(N²) → O(N) optimization
- **100% compatibility** with all `cli_tts.py` arguments and features
- **Real-time memory monitoring** with detailed performance metrics
- **Multiple backend fallback** for maximum stability
- **Enhanced MPS support** for Apple Silicon with aggressive memory management
- **Best performance on longer texts** and batch generation

**⚡ Flash-Specific Options:**
- `--monitor` - Show detailed memory and performance monitoring
- `--no-flash` - Disable Flash Attention (for comparison)
- `--benchmark` - Run Flash Attention performance benchmark

**📖 Complete Documentation:** See [FLASH_CLI_USAGE.md](FLASH_CLI_USAGE.md) for comprehensive guide with examples and troubleshooting.

**💡 Recommendation:** Use `flash_optimized_cli.py` as your default CLI - it provides all the same functionality as `cli_tts.py` plus significant performance improvements!

### Model Versions

Fish Speech supports different model versions with varying capabilities:

```bash
# Use Fish Speech 1.5 (default)
poetry run python cli_tts.py "Hello world" --model-version 1.5

# Use Fish Speech 1.6 (latest with more features)
poetry run python cli_tts.py "Hello world" --model-version 1.6

# Use Fish Speech 1.4 (older version)
poetry run python cli_tts.py "Hello world" --model-version 1.4
```

**Model Options:**
- `--model-version` - Model version: 1.4, 1.5, 1.6 (default: 1.5)

### Voice Control and Prosody

Control speech characteristics with advanced parameters:

```bash
# Speed control (0.5x to 2.0x)
poetry run python cli_tts.py "Fast speech" --speed 1.5
poetry run python cli_tts.py "Slow speech" --speed 0.7

# Volume control (-20dB to +20dB)
poetry run python cli_tts.py "Loud speech" --volume 5
poetry run python cli_tts.py "Quiet speech" --volume -5

# Pitch control (0.5x to 2.0x)
poetry run python cli_tts.py "High pitch" --pitch 1.3
poetry run python cli_tts.py "Low pitch" --pitch 0.8

# Combine multiple parameters
poetry run python cli_tts.py "Custom speech" --speed 1.2 --volume 3 --pitch 0.9
```

**Prosody Options:**
- `--speed FLOAT` - Speech speed (0.5-2.0, default: 1.0)
- `--volume INT` - Volume in dB (-20 to +20, default: 0)
- `--pitch FLOAT` - Pitch adjustment (0.5-2.0, default: 1.0)

**Note:** Prosody effects (speed, volume, pitch) are applied as post-processing after audio generation using scipy signal processing.

### Future Features (Coming Soon)

The following emotion and style parameters are included in the CLI but not yet fully supported by the current Fish Speech model:

```bash
# Emotional speech (experimental - may not work in current version)
poetry run python cli_tts.py "I'm so happy!" --emotion happy --intensity 0.8
poetry run python cli_tts.py "This is sad news" --emotion sad --intensity 0.6

# Speaking styles (experimental - may not work in current version)
poetry run python cli_tts.py "Ladies and gentlemen" --style formal
poetry run python cli_tts.py "Hey there, buddy!" --style casual
```

**Emotion and Style Options (Experimental):**
- `--emotion` - Emotion: happy, sad, angry, neutral, excited
- `--intensity FLOAT` - Emotion intensity (0.0-1.0, default: 0.5)
- `--style` - Speaking style: formal, casual, dramatic

*These features may become available in future versions of Fish Speech or require specialized model checkpoints.*

### Voice Cloning with Reference Audio

Fish Speech supports voice cloning using reference audio samples:

#### Step 1: Create Voice Reference
```bash
# Create a voice reference from your audio file
poetry run python cli_tts.py --create-reference your_voice.wav voices/your_voice_ref.npy
```

#### Step 2: Use Voice Reference for Synthesis
```bash
# Synthesize with your cloned voice (direct text)
poetry run python cli_tts.py "Text to speak" \
  --prompt-tokens voices/your_voice_ref.npy \
  --prompt-text "Original text from reference audio" \
  -o output/cloned_voice.wav

# Synthesize with your cloned voice (text from file)
poetry run python cli_tts.py "Text to speak" \
  --prompt-tokens voices/your_voice_ref.npy \
  --prompt-tokens-file voices/reference_text.txt \
  -o output/cloned_voice.wav
```

#### Voice Management
```bash
# List all available voices
poetry run python cli_tts.py --list-voices

# Use a voice by name (automatically finds .npy and .txt files)
poetry run python cli_tts.py "Hello world" --voice RU_Male_Deadpool

# Combine voice cloning with prosody
poetry run python cli_tts.py "Fast speech with prosody" \
  --voice RU_Male_Deadpool \
  --speed 1.3 --volume 3 --pitch 1.1

# Advanced example: Custom prosody with voice cloning
poetry run python cli_tts.py "I'm excited to announce this news!" \
  --voice RU_Male_Deadpool \
  --speed 1.2 --volume 3 \
  --model-version 1.6 \
  -o output/announcement.wav --play

# Prosody example: Slow, quiet presentation style
poetry run python cli_tts.py "Ladies and gentlemen, welcome to our presentation" \
  --speed 0.9 --volume -2 \
  --model-version 1.6

# High-pitched excited speech
poetry run python cli_tts.py "This is amazing news!" \
  --speed 1.4 --volume 5 --pitch 1.3 \
  -o output/excited.wav --play
```

**Voice Cloning Options:**
- `--create-reference audio_file output.npy` - Create voice reference from audio
- `--prompt-tokens file.npy` - Use voice reference for synthesis
- `--prompt-text "text"` - Text corresponding to reference audio (**required** with --prompt-tokens)
- `--prompt-tokens-file file.txt` - Load reference text from file (alternative to --prompt-text)
- `--voice VOICE_NAME` - Use voice by name from voices/ folder
- `--list-voices` - Show all available voices

### Caching System

Semantic token caching speeds up repeated requests:

```bash
# Normal usage with caching enabled (default)
poetry run python cli_tts.py "Hello world"

# Disable caching for a specific request
poetry run python cli_tts.py "Hello world" --no-cache

# View cache information
poetry run python cli_tts.py --cache-info

# Clear semantic tokens cache
poetry run python cli_tts.py --clear-cache
```

**Cache Options:**
- `--no-cache` - Disable semantic tokens caching
- `--cache-info` - Show cache size and statistics
- `--clear-cache` - Clear semantic tokens cache

## Fine-tuning Custom Voices

Fish Speech supports fine-tuning to create custom voice models trained on your own data. This allows for more accurate voice cloning and personalized TTS models.

### Quick Start

```bash
# 1. Prepare your training data (audio + text files)
poetry run python prepare_dataset.py \
  --input my_voice_recordings/ \
  --output training_data/my_voice \
  --auto-transcribe --normalize --split-long

# 2. Run complete fine-tuning pipeline
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_voice \
  --full-pipeline \
  --max-steps 1000 \
  --batch-size 4
```

### Dataset Preparation

#### From Local Audio Files
```bash
# Process local directory with automatic transcription
poetry run python prepare_dataset.py \
  --input raw_audio_folder/ \
  --output training_data/processed \
  --normalize \
  --split-long \
  --auto-transcribe \
  --whisper-model medium

# Manual dataset structure
training_data/
└── my_voice/
    ├── SPEAKER_NAME/
    │   ├── audio_001.wav  # 10-30 second clips
    │   ├── audio_001.lab  # exact transcription
    │   ├── audio_002.wav
    │   ├── audio_002.lab
    │   └── ...
    └── dataset_summary.json
```

#### From YouTube Videos
```bash
# Download and process YouTube content with automatic transcription
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model large
```

### Fine-tuning Process

#### Full Automated Pipeline
```bash
# Complete training process in one command
poetry run python finetune_tts.py \
  --project my_voice_model \
  --data-dir training_data/my_voice \
  --model-version 1.5 \
  --full-pipeline \
  --max-steps 1500 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --device mps  # Use Apple Silicon GPU
```

#### Step-by-Step Process
```bash
# Step 1: Extract semantic tokens from audio
poetry run python finetune_tts.py \
  --project my_voice \
  --data-dir training_data/my_voice \
  --extract-tokens \
  --batch-size-extract 16

# Step 2: Train the model with LoRA
poetry run python finetune_tts.py \
  --project my_voice \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 1000 \
  --batch-size 2

# Step 3: Merge LoRA weights with base model
poetry run python finetune_tts.py \
  --project my_voice \
  --merge-weights
```

### Training Parameters

| Parameter | Description | Recommendations |
|-----------|-------------|------------------|
| `--max-steps` | Training steps | 1000-2000 for good results |
| `--batch-size` | Batch size | 2-4 (depends on GPU memory) |
| `--learning-rate` | Learning rate | 1e-4 to 5e-5 |
| `--model-version` | Base model | 1.5 (stable) or 1.6 (latest) |
| `--device` | Device | mps (Apple Silicon), cuda (NVIDIA) |

### Data Requirements

- **Minimum**: 10-30 minutes of high-quality audio
- **Recommended**: 30-60 minutes of diverse content
- **Audio Quality**: 44.1kHz, mono, minimal background noise
- **Text**: Accurate transcriptions for all audio
- **Content**: Varied speaking styles and vocabulary

### Using Fine-tuned Models

After fine-tuning, your custom model will be saved to:
```
checkpoints/my_voice_model-merged/
├── text2semantic_500M.pth
├── tokenizer.tiktoken
└── ...
```

Integration with CLI TTS:
```python
# Add to cli_tts.py setup_fish_speech() function:
custom_model_path = "checkpoints/my_voice_model-merged"
if Path(custom_model_path).exists():
    return fish_speech_dir, Path(custom_model_path)
```

### Examples

```bash
# Train on podcast recordings
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=PODCAST_ID" \
  --output training_data/podcast_host \
  --speaker "PodcastHost" --youtube --auto-transcribe

poetry run python finetune_tts.py \
  --project podcast_voice \
  --data-dir training_data/podcast_host \
  --full-pipeline --max-steps 1500

# Train on audiobook narration
poetry run python prepare_dataset.py \
  --input audiobook_chapters/ \
  --output training_data/narrator \
  --auto-transcribe --split-long

poetry run python finetune_tts.py \
  --project audiobook_narrator \
  --data-dir training_data/narrator \
  --full-pipeline --max-steps 2000 --learning-rate 2e-5

# Quick test with small dataset
poetry run python finetune_tts.py \
  --project quick_test \
  --data-dir small_dataset/ \
  --full-pipeline --max-steps 500 --batch-size 2
```

For detailed fine-tuning instructions, see [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md).

### Web Interface

For interactive use with reference audio:

```bash
poetry run python simple_tts.py
```

This launches the Fish Speech WebUI at http://localhost:7860

## Examples

```bash
# 🔥 Flash-Optimized CLI Examples (Recommended)

# Generate speech with enhanced performance
poetry run python flash_optimized_cli.py "Hello, this is Flash-Optimized Fish Speech!"

# Long text with memory monitoring
poetry run python flash_optimized_cli.py "This is a longer text that benefits from Flash Attention optimizations for better memory efficiency and faster processing" --monitor

# Voice cloning with Flash optimizations
poetry run python flash_optimized_cli.py "Flash-optimized voice cloning example" \
  --voice RU_Male_Deadpool --monitor --play

# Performance benchmark
poetry run python flash_optimized_cli.py --benchmark

# Standard CLI Examples

# Generate speech in English
poetry run python cli_tts.py "Hello, this is Fish Speech running on Apple Silicon"

# Generate speech in Russian  
poetry run python cli_tts.py "Привет! Это Fish Speech на Apple Silicon"

# Generate and play immediately
poetry run python cli_tts.py "Testing TTS synthesis" --play

# Save to custom file in output directory
poetry run python cli_tts.py "Custom output example" -o output/my_speech.wav

# Create voice reference from audio
poetry run python cli_tts.py --create-reference sample_voice.wav voices/my_voice.npy

# Clone voice with reference
poetry run python cli_tts.py "This will sound like the reference voice" \
  --prompt-tokens voices/my_voice.npy \
  --prompt-text "Original reference text" \
  -o output/cloned_speech.wav --play

# Clone voice with reference text from file
poetry run python cli_tts.py "Text from file example" \
  --prompt-tokens voices/my_voice.npy \
  --prompt-tokens-file voices/reference_text.txt \
  -o output/cloned_speech_file.wav --play

# Advanced example: Emotional speech with custom voice and prosody
poetry run python cli_tts.py "I'm so excited to announce this news!" \
  --voice RU_Male_Deadpool \
  --emotion excited --intensity 0.9 \
  --speed 1.2 --volume 3 \
  --model-version 1.6 \
  -o output/announcement.wav --play

# Formal presentation style
poetry run python cli_tts.py "Ladies and gentlemen, welcome to our presentation" \
  --style formal --speed 0.9 --volume 2 \
  --model-version 1.6

# Show model cache information
poetry run python cli_tts.py --cache-info
```

**Important:** When using `--prompt-tokens`, you **must** also specify either `--prompt-text` or `--prompt-tokens-file` with the text that corresponds to your reference audio.

## Project Structure

```
fs-python/
├── cli_tts.py              # Main CLI interface
├── flash_optimized_cli.py  # Flash-Optimized CLI with enhanced performance
├── simple_tts.py           # Web interface
├── finetune_tts.py         # Fine-tuning pipeline
├── prepare_dataset.py      # Dataset preparation
├── voices/                 # Voice reference files (.npy and .txt)
├── output/                 # Generated audio files
├── cache/                  # Semantic tokens cache
├── training_data/          # Fine-tuning datasets
├── checkpoints/            # Custom trained models
├── fish-speech/            # External Fish Speech repository
├── pyproject.toml          # Poetry dependencies
├── README.md               # This file
├── FLASH_CLI_USAGE.md      # Flash-Optimized CLI complete guide
├── FINE_TUNING_GUIDE.md    # Detailed fine-tuning guide
└── .gitignore              # Git ignore rules
```

## Model Storage

The first run will automatically download the Fish Speech model (~1.3GB) to the Hugging Face cache (`~/.cache/huggingface/hub/`) rather than a local directory. This approach:
- Saves disk space by avoiding duplicate model downloads
- Follows standard Hugging Face conventions  
- Allows sharing models between different projects
- Automatically handles model updates and versioning

## Performance

- **Fish Speech 1.5**: Balanced performance and quality
- **Fish Speech 1.6**: Latest version with improved emotional control
- **Generation Speed**: ~20 tokens/second on Apple Silicon
- **Real-time Factor**: ~1:5 on M1/M2, faster on M3/M4
- **Memory Usage**: ~4GB for inference, ~8GB for fine-tuning
- **Cache Benefits**: 2-5x speedup for repeated text with same voice
- **Fine-tuning Time**: 1-4 hours (depends on dataset size and hardware)

## Troubleshooting

### Common Issues

1. **MPS not available**: The script automatically falls back to CPU
2. **Model download fails**: Check internet connection and HuggingFace access
3. **Audio playback issues**: Install `afplay` (usually available on macOS)
4. **Out of memory**: Try using `--device cpu` or close other applications
5. **Fine-tuning OOM**: Reduce `--batch-size` or use gradient accumulation

### Error Messages

- `❌ Fish Speech не найден`: Fish Speech уже включен в проект, убедитесь что он установлен
- `❌ Файл голоса не найден`: Check voice file path in `voices/` folder
- `❌ При использовании --prompt-tokens нужно указать --prompt-text`: Provide reference text
- `❌ Ошибка извлечения токенов`: Check audio file format and GPU memory
- `❌ Ошибка обучения`: Verify dataset structure and training parameters

### Fine-tuning Troubleshooting

- **Poor quality results**: Increase training steps or improve dataset quality
- **Training too slow**: Use smaller batch size or check GPU utilization
- **Model not converging**: Adjust learning rate or increase dataset size
- **Memory issues**: Reduce batch size, use CPU, or increase system RAM

## License

This project uses Fish Speech under CC-BY-NC-SA-4.0 license. Please respect the licensing terms when using for commercial purposes.

## Checkpoint Inference

### Automatic LoRA Checkpoint Conversion

The CLI now supports automatic conversion and inference with Lightning LoRA checkpoints:

```bash
# Basic checkpoint inference
python cli_tts.py "Your text here" --checkpoint path/to/checkpoint.ckpt

# Flash-optimized checkpoint inference  
python flash_optimized_cli.py "Your text here" --checkpoint path/to/checkpoint.ckpt
```

### How it works:

1. **Automatic Detection**: System detects Lightning checkpoint files (.ckpt)
2. **Config Parsing**: Reads hydra config to extract base model and LoRA parameters
3. **Automatic Conversion**: Converts LoRA checkpoint to inference format using `merge_lora.py`
4. **Caching**: Converted models are cached for future use
5. **Inference**: Uses the merged model for generation

### Supported Checkpoint Types:

- **Lightning LoRA Checkpoints** (.ckpt files) - Automatically converted
- **Inference Model Directories** - Used directly
- **Custom Model Paths** - Specified with --model-path

### Example Usage:

```bash
# Compare base model vs LoRA checkpoint
python cli_tts.py "Test text" -o base_model.wav
python cli_tts.py "Test text" --checkpoint checkpoints/my_model/step_001000.ckpt -o lora_model.wav

# Flash-optimized with checkpoint
python flash_optimized_cli.py "Long text for generation" \
  --checkpoint checkpoints/my_model/step_001000.ckpt \
  --monitor
```

### Cache Management:

Converted checkpoints are cached in the same directory as the original checkpoint:
```
checkpoints/my_model/
├── checkpoints/
│   ├── step_001000.ckpt
│   └── converted_inference/     # Auto-generated cache
│       ├── model.pth
│       ├── config.json
│       └── ...
└── .hydra/
    └── config.yaml
```