# Environment Variables Setup

## Quick Creation of .env File

Use the provided script to create a template:

```bash
# Create .env.example with full configuration
./create-env-example.sh

# Copy and customize for your needs
cp .env.example .env
# Edit the .env file and set your API keys
```

## Minimal Configuration

Create a `.env` file **in the project root**:

```env
# AI API keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Server settings
PORT=3000
CONCURRENT_REQUESTS=3

# Frontend
VITE_API_URL=http://localhost:3000
```

## Advanced Gemini API Configuration

For fine-tuning the AI model, additional parameters are available:

```env
# Model and quality
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_OUTPUT_TOKENS=8192

# Performance
GEMINI_TIMEOUT=120000
GEMINI_RETRY_COUNT=3

# Content safety
GEMINI_HARASSMENT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_HATE_SPEECH_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
# ... and other parameters
```

**📚 Full configuration guide**: [docs/GEMINI_CONFIG.md](./GEMINI_CONFIG.md)

This file contains:
- All available parameters with descriptions
- Recommended settings for different scenarios
- Example configurations for various purposes
- Security and performance settings

**📁 Important**: Create the `.env` file in the project root, not in the `backend/` or `frontend/` folders. 