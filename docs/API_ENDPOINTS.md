# API Endpoints

## GET /status
Checks server status
```json
{
  "status": "ok",
  "message": "Podcast server is running",
  "endpoints": {...}
}
```

## POST /generate-script
Generates a podcast script
```json
{
  "userPrompt": "Podcast topic...",
  "systemPrompt": "Optional prompt..."
}
```

## POST /generate-podcast
Generates podcast audio (with WebSocket progress)
```json
{
  "conversationData": {...},        // JSON data
  "conversationFile": "file.json"   // or file path
}
```

## GET /podcasts
Returns a list of created podcasts
```json
{
  "podcasts": [
    {
      "filename": "podcast.wav",
      "url": "/output/podcast.wav",
      "size": 1024000,
      "created": "2024-01-01T00:00:00Z",
      "modified": "2024-01-01T00:00:00Z"
    }
  ]
}
```

## WebSocket /
Real-time progress:
```json
{
  "type": "progress",
  "jobId": "1640995200000",
  "current": 25,
  "total": 50,
  "message": "Generating audio 25/50"
}
``` 