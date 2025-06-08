# Worker-Based Audio Generation System

## Overview

The new audio generation system uses long-lived Python workers instead of creating a new process for each voice line. This significantly speeds up processing since the model is loaded only once.

## Architecture

```
[Node.js Backend]
    ↓
[WorkerManager] → [AudioWorker 0] → [Python Process 0]
                → [AudioWorker 1] → [Python Process 1]  
                → [AudioWorker N] → [Python Process N]
```

### Main Components:

1. **`WorkerManager`** (TypeScript) - manages worker pool
2. **`AudioWorker`** (TypeScript) - manages individual Python process
3. **`audio_worker.py`** (Python) - long-lived process for audio generation

## Configuration

📖 **Complete environment variables documentation:** [docs/ENV_SETUP.md](docs/ENV_SETUP.md)

### Main Variables

- `CONCURRENT_REQUESTS` - number of workers (default: 2)
- `AUDIO_DEVICE` - processing device (mps/cuda/cpu)
- `MODEL_VERSION` - Fish Speech model version (default: 1.5)
- `SEMANTIC_TOKEN_CACHE` - token caching (default: true)

### Creating .env File

```bash
cp backend/config/worker.config.example .env
# Edit values as needed
```

## Usage

### Basic Usage

```typescript
import { generateAudio, processMessageChunks } from "./helpers/audioGenerator.js";

// Generate single audio
const filePath = await generateAudio(
  "Hello, world!", 
  "default", 
  1, 
  1, 
  "output"
);

// Process array of messages
const messages = [
  { text: "First message", speaker: "default", id: 1 },
  { text: "Second message", speaker: "default", id: 2 }
];

const chunks = createMessageChunks(messages, 10);
const generatedFiles = await processMessageChunks(chunks, "output");
```

### Worker Management

```typescript
import { healthCheck, getWorkerStatus, shutdownWorkerManager } from "./helpers/audioGenerator.js";

// Health check
const isHealthy = await healthCheck();

// Get worker status
const status = getWorkerStatus();
console.log(status);

// Force shutdown (for testing)
shutdownWorkerManager();
```

## Advantages of the New System

### Performance
- **Model loads only once** instead of loading for each voice line
- **Parallel processing** with N workers simultaneously
- **Semantic token caching** between requests (configurable via `SEMANTIC_TOKEN_CACHE`)
- **Memory optimization** with Flash Attention

### Reliability
- **Automatic recovery** when worker crashes
- **Health checks** for monitoring status
- **Graceful shutdown** with proper task completion
- **Timeouts** to prevent hanging requests

### Monitoring
- **Detailed statistics** on memory usage and generation time
- **Request queue** with status tracking
- **Colored output** for easy process tracking

## Testing

### Running Tests

```bash
# From backend directory
npm run test:worker

# Or direct run
npx tsx test/testWorkerManager.ts
```

### Manual Testing

```bash
# Run individual worker for debugging
cd fs-python
poetry run python audio_worker.py 0 mps

# Send test request
echo '{"type":"load_model","model_version":"1.5"}' | poetry run python audio_worker.py 0
echo '{"type":"generate","id":1,"text":"Test","output_path":"test.wav","voice_settings":{}}' | poetry run python audio_worker.py 0
```

## Migration

### Before (old system)

```typescript
// Each call created a new Python process
const filePath = await generateAudio(text, speaker, id, streamIndex, outputDir);
```

### After (new system)

```typescript
// Workers initialize once and are reused
const filePath = await generateAudio(text, speaker, id, streamIndex, outputDir);
// API remains the same but works significantly faster
```

## Troubleshooting

📖 **Complete guide:** [docs/ENV_SETUP.md#-troubleshooting](docs/ENV_SETUP.md#-troubleshooting)

## Performance Monitoring

### Key Metrics

- **Initialization time**: worker startup time
- **Generation time**: single audio creation time
- **Throughput**: voice lines per second
- **Memory usage**: RAM and GPU memory per worker
- **Queue size**: number of pending requests

### Typical Performance Values

- **Initialization**: 30-60 seconds (model loading)
- **Generation**: 5-15 seconds per voice line (depending on length)
- **Memory**: 3-6 GB RAM + 2-4 GB GPU per worker
- **Speedup**: 2-5x compared to old system

## Limitations

1. **Initial initialization** takes more time
2. **Memory consumption** higher due to loaded models
3. **Number of workers** limited by available memory
4. **Custom models** must be compatible with Fish Speech 1.5+

## Roadmap

- [ ] Automatic worker scaling
- [ ] Distributed processing across multiple machines
- [ ] Web interface for monitoring
- [ ] Integration with queue systems (Redis/RabbitMQ)
- [ ] Hot-reload model support 