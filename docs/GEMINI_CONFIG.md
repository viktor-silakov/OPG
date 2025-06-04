# Gemini API Configuration - .env Parameters

This file contains all available parameters for configuring the Gemini API in the **root** `.env` file of your project.

## 📁 Configuration Placement

Create a `.env` file in the **root of your project** (not in `backend/.env`):

```
OpenNotebookLmRU.rs/
├── .env                     # ← Here! (root configuration)
├── backend/
├── frontend/
├── prompts/
└── ...
```

## 🔑 Basic Parameters

```bash
# Required API key
GEMINI_API_KEY=your_gemini_api_key_here

# Alternative AI (optional)
OPENAI_API_KEY=your_openai_api_key_here
```

## 🤖 Gemini Model Parameters

```bash
# Model to use
GEMINI_MODEL=gemini-2.0-flash
# Available options:
# - gemini-1.5-pro-latest (recommended)
# - gemini-1.5-pro
# - gemini-1.5-flash (fast)
# - gemini-pro (legacy)

# Temperature (creativity) - from 0.0 to 2.0
GEMINI_TEMPERATURE=0.7
# 0.0 = deterministic, predictable
# 0.7 = balanced (recommended)
# 1.0 = creative
# 2.0 = maximum creativity

# Maximum output tokens
GEMINI_MAX_OUTPUT_TOKENS=65536
# Model limits:
# - gemini-1.5-pro: up to 8192
# - gemini-1.5-flash: up to 8192
# - gemini-pro: up to 2048

# Top-p (nucleus sampling) - from 0.0 to 1.0
GEMINI_TOP_P=0.95
# Controls response diversity
# 0.8-0.95 is the recommended range

# Top-k sampling - number of top tokens
GEMINI_TOP_K=40
# Range: 1-40
# 20-40 is the recommended range

# Number of response candidates
GEMINI_CANDIDATE_COUNT=1
# Usually 1, higher values slow down responses
```

## 🔒 Safety Settings

```bash
# Content filtering levels
# BLOCK_NONE - no filtering
# BLOCK_ONLY_HIGH - only high risk
# BLOCK_MEDIUM_AND_ABOVE - medium and high (recommended)
# BLOCK_LOW_AND_ABOVE - all levels

GEMINI_HARASSMENT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_HATE_SPEECH_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_SEXUALLY_EXPLICIT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_DANGEROUS_CONTENT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
```

## ⚡ Performance Parameters

```bash
# Request timeout in milliseconds
GEMINI_TIMEOUT=120000
# 120 seconds for complex podcast generation requests

# Number of retries on error
GEMINI_RETRY_COUNT=3
# 2-3 retries recommended

# Delay between retries (ms)
GEMINI_RETRY_DELAY=2000
# 2 seconds between retries
```

## 🚀 OpenAI as fallback

```bash
# OpenAI model (if used as fallback)
OPENAI_MODEL=gpt-4-turbo-preview
# Alternatives: gpt-4, gpt-3.5-turbo

# OpenAI parameters
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4096
```

## 🛠️ System Settings

```bash
# Main server settings
PORT=3000
CONCURRENT_REQUESTS=3
MAX_REQUEST_SIZE=50mb

# Fish Speech settings
FISH_SPEECH_URL=http://localhost:7860
FISH_SPEECH_TIMEOUT=300000
FISH_SPEECH_RETRY_COUNT=2

# File settings
TEMP_FILE_TTL=3600000
AUTO_CLEANUP_TEMP_FILES=true
MAX_UPLOAD_SIZE=100

# Logging
LOG_LEVEL=info
DEBUG_AI_REQUESTS=false
DEBUG_WEBSOCKET=false
SAVE_LOGS_TO_FILE=false

# Security
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
SECURE_COOKIES=false
SESSION_SECRET=your_random_session_secret_here

# Caching
ENABLE_AI_CACHE=true
AI_CACHE_TTL=1800
AI_CACHE_MAX_SIZE=1000
CONNECTION_POOL_SIZE=10
```

## 📋 Example of a complete .env file

Create a `.env` file **in the project root** with the following content:

```bash
# ============================================
# AI API KEYS
# ============================================
GEMINI_API_KEY=your_actual_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# ============================================
# GEMINI SETTINGS
# ============================================
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_OUTPUT_TOKENS=65536
GEMINI_TOP_P=0.95
GEMINI_TOP_K=40
GEMINI_CANDIDATE_COUNT=1
GEMINI_TIMEOUT=120000
GEMINI_RETRY_COUNT=3
GEMINI_RETRY_DELAY=2000

# ============================================
# SAFETY
# ============================================
GEMINI_HARASSMENT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_HATE_SPEECH_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_SEXUALLY_EXPLICIT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
GEMINI_DANGEROUS_CONTENT_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE

# ============================================
# SERVER
# ============================================
PORT=3000
CONCURRENT_REQUESTS=3
FISH_SPEECH_URL=http://localhost:7860
LOG_LEVEL=info

# ============================================
# FRONTEND
# ============================================
VITE_API_URL=http://localhost:3000
```

## 🎯 Recommended settings

### For high-quality podcasts:
```bash
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_OUTPUT_TOKENS=65536
GEMINI_TOP_P=0.9
GEMINI_TOP_K=40
```

### For fast generation:
```bash
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.5
GEMINI_MAX_OUTPUT_TOKENS=4096
GEMINI_TOP_P=0.8
GEMINI_TOP_K=20
```

### For maximum creativity:
```bash
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_TEMPERATURE=1.2
GEMINI_MAX_OUTPUT_TOKENS=65536
GEMINI_TOP_P=0.95
GEMINI_TOP_K=40
```

### For conservative content:
```bash
GEMINI_HARASSMENT_THRESHOLD=BLOCK_LOW_AND_ABOVE
GEMINI_HATE_SPEECH_THRESHOLD=BLOCK_LOW_AND_ABOVE
GEMINI_SEXUALLY_EXPLICIT_THRESHOLD=BLOCK_LOW_AND_ABOVE
GEMINI_DANGEROUS_CONTENT_THRESHOLD=BLOCK_LOW_AND_ABOVE
```

## 🔍 Monitoring and debugging

To debug AI requests, enable:
```bash
LOG_LEVEL=debug
DEBUG_AI_REQUESTS=true
```

To debug WebSocket:
```bash
DEBUG_WEBSOCKET=true
```

## 📊 Performance optimization

For high load:
```bash
CONCURRENT_REQUESTS=5
CONNECTION_POOL_SIZE=15
ENABLE_AI_CACHE=true
AI_CACHE_TTL=3600
AI_CACHE_MAX_SIZE=2000
```

For resource saving:
```bash
CONCURRENT_REQUESTS=2
CONNECTION_POOL_SIZE=5
ENABLE_AI_CACHE=true
AI_CACHE_TTL=1200
AI_CACHE_MAX_SIZE=500
```

## ⚠️ Important notes

1. **API keys**: Never commit real keys to git
2. **Limits**: Watch your Gemini API account limits
3. **Temperature**: High values may produce unpredictable results
4. **Timeouts**: Too short timeouts may interrupt complex requests
5. **Caching**: Enable to save API calls

## 🚀 Applying changes

After changing the `.env` file:
1. Stop the server (Ctrl+C)
2. Restart: `npm run server`
3. Check logs for errors 