# Technical Details

## Fish Speech TTS
- Uses Fish Speech 1.5/1.6 models
- MPS acceleration support on Apple Silicon
- Semantic token caching
- Voice cloning with reference audio

## Voice Management
- Automatic token creation via `cli_tts.py --create-reference`
- Supports WAV files up to 50MB
- File format validation
- Safe file deletion

## API and WebSocket
- RESTful API for all operations
- WebSocket for real-time progress
- Error handling and validation
- Automatic cleanup of temporary files 