# Open Podcast Generator (NotebookLM Alternative)

An open version of NotebookLM for podcast generation using Fish Speech TTS and Apple Silicon (MPS) acceleration.

## Features

- 🎙️ **Podcast Generation** - Create audio podcasts from text scripts using Fish Speech TTS
- 🍎 **Apple Silicon Support** - for audio inference and fine-tuning of Fish Speech models
- ⚡ **Flash Attention Optimization** - Enhanced memory-efficient attention with O(N²) → O(N) complexity reduction, 30-50% memory savings during TTS inference
- 🧠 **MPS Acceleration** - Metal Performance Shaders optimization for Apple Silicon with automatic backend fallbacks and real-time memory management
- 📝 **Script Creation** - Automatic generation of dialogues using Gemini AI, including free models
- 🎭 **Voice Management** - Upload and create your own zero-shot voices, more than 80 voices in various languages (🇸🇦 Arabic, 🇧🇾 Belarusian, 🇩🇪 German, 🇺🇸 English, 🇪🇸 Spanish, 🇫🇷 French, 🇬🇪 Georgian, 🇯🇵 Japanese, 🇰🇷 Korean, 🇷🇺 Russian, 🇺🇦 Ukrainian) and styles are built-in by default
- 📚 **Podcast Library** - View and manage created podcasts
- 🎨 **Podcast Style Management** - Customize various hosting styles through prompts
- 🎛️ **Voice Fine-Tuning** - Adjust speed, volume, tempo, and fine-tune specific voices
- 📊 **Data Preparation for Fine-Tuning** - Automatic processing of YouTube videos, audiobooks, and your own voice
- 🔧 **Fish-Speech Model Fine-Tuning** - Train custom voice models (see [fs-python](./fs-python/README.md))
- 🚀 **Custom TTS Inference** - Full control over the speech synthesis process (see [fs-python](./fs-python/README.md))

## 📋 Prerequisites

Before starting, make sure you have the following installed on your system:

### System Requirements

- **Operating System**: macOS 11.0+ (tested on macOS Sequoia 15.3+) with Apple Silicon
- **Hardware**: Apple M-series chip (M1/M2/M3+) for optimal AI inference performance

### Required Software

#### Node.js & Package Managers

- **Node.js**: tested with v22.4.0
- **Yarn**: ≥1.22.0

#### Python Environment

- **Python**: 3.11+ (tested with 3.11.12)
- **Poetry**: ≥1.7.0 for Python dependency management

  ```bash
  # Install Poetry if not installed
  curl -sSL https://install.python-poetry.org | python3 -
  ```

#### System Tools

- **Git**: for repository management
- **FFmpeg**: for audio processing and format conversion
- **System Dependencies**: Required for Fish Speech compilation

  ```bash
  # Install via Homebrew
  brew install ffmpeg cmake pkg-config protobuf sentencepiece
  ```

### AI Model Requirements

- **Gemini API Key**: Required for script generation (supports free tier)
- **Minimum 16GB RAM**: Recommended for running Fish Speech models
- **10GB+ free disk space**: For models, voices, and generated content

## 🚀 Quick Start

### 1. Install Fish Speech (fs-python)

```bash
# Navigate to Fish Speech directory
cd fs-python

# Install Poetry dependencies
poetry install

# Set environment variables for compilation (macOS)
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"

# Install Fish Speech in editable mode
poetry run pip install -e ./fish-speech

# Additional dependencies for fine-tuning (optional)
poetry run pip install librosa soundfile whisper tqdm yt-dlp

# Test installation
poetry run python -c "import fish_speech; print('Fish Speech successfully installed!')"

cd ..
```

#### Troubleshooting Fish Speech Installation

If you encounter compilation errors:

1. **sentencepiece build failure**: Install system dependencies first
   ```bash
   brew install cmake pkg-config protobuf sentencepiece
   export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
   ```

2. **CMake version errors**: Update CMake to latest version
   ```bash
   brew upgrade cmake
   ```

### 2. Install Dependencies

```bash
# Install all dependencies with one command
yarn run install-all

# Or separately:
# Backend dependencies
cd backend && yarn install && cd ..

# Frontend dependencies  
cd frontend && yarn install && cd ..
```

### 3. Environment Variables Setup

Create a `.env` file in the root directory.

```
touch .env
```

Set `GEMINI_API_KEY` in .env file and other variables as needed.

See the full environment variables setup guide here: [docs/ENV_SETUP.md](./docs/ENV_SETUP.md)

### 4. System Launch

#### Recommended way - launch the entire system

```bash
yarn start
```

This command will launch:

- **Backend server** at <http://localhost:3000>
- **Frontend application** at <http://localhost:5173>

## 🎨 Available Podcast Styles

In the `prompts/` folder, there are various system prompts for different hosting styles:

- **business** - business-professional style with corporate expertise
- **comedy** - comedy-entertainment style with witty humor
- **debate** - discussion-analytical style with clashing opinions
- **default** - friendly-informative style with light humor
- **goblin** - cynical-energetic style in the spirit of Dmitry Puchkov  
- **scientific** - scientific-educational academic style
- **storytelling** - narrative-captivating style with vivid stories

Each prompt contains:

- Unique host voices with characteristic names
- Specific phrases and stylistic techniques
- Characteristic tone of narration and presentation of material
- Dialogue samples corresponding to the style

**Automatic detection:** Any `.md` file added to the `prompts/` folder will automatically appear in the style selection dropdown without restarting the server.

## ➕ Adding New Podcast Styles

To add a new podcast style:

1. Create a new `.md` file in the `prompts/` folder with any name
2. Use existing prompts as a template  
3. Customize the prompt content for your style
4. The file **will automatically appear** in the dropdown without restarting the server

**Example:** Create `prompts/comedy.md` → the "comedy" option will appear in the dropdown

## 📋 Key Commands

Most commands are listed in [docs/COMMANDS.md](./docs/COMMANDS.md). Main commands:

```bash
yarn start         # Start backend + frontend
yarn server    # Start backend only
yarn frontend  # Start frontend only
yarn build     # Build frontend for production
```

### How to Create a New Voice

1. Go to the **"Voice Management"** tab
2. Click the **"Add Voice"** button
3. Fill out the form:
   - **Voice Name** - unique name (e.g., `RU_Male_MyVoice`)
   - **Audio File** - WAV file with a speech sample (10-30 seconds)
   - **Reference Text** - exact transcription of the audio file
4. Click **"Create Voice"**

The system will automatically:

- Save the audio file in `fs-python/voices/`
- Create a text file with the transcription
- Generate voice tokens using Fish Speech
- Add the voice to the list of available voices

### Voice File Structure

```
fs-python/voices/
├── RU_Male_MyVoice.wav    # Audio file
├── RU_Male_MyVoice.txt    # Text transcription  
└── RU_Male_MyVoice.npy    # Voice tokens
```

## Documentation

See [docs/](./docs/) for more information.

## 📄 License

MIT for the project code, but note that this project uses Fish Speech under the CC-BY-NC-SA-4.0 license. Please comply with the licensing terms for commercial use.
