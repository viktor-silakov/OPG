# Project Architecture

```
OpenNotebookLmRU.rs/
├── frontend/              # React + TypeScript + Mantine UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── VoiceManager.tsx    # Voice management
│   │   │   ├── PodcastWorkflow.tsx # Podcast creation
│   │   │   ├── PodcastLibrary.tsx  # Podcast library
│   │   │   └── StatusChecker.tsx   # Server status
│   │   └── api/client.ts           # API client
├── backend/               # Node.js + Express + TypeScript
│   ├── server.ts          # Main server with voice endpoints
│   └── helpers/           # Helper functions
├── fs-python/             # Fish Speech TTS
│   ├── cli_tts.py         # CLI for speech synthesis
│   ├── voices/            # Voices directory
│   └── output/            # Generated audio files
├── prompts/               # System prompts for AI
└── output/                # Final podcasts
``` 