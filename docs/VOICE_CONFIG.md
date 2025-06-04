# Voice Configuration

This file describes how to set up voice parameters for podcast generation.

## Configuration File

The main configuration is in the `voice-config.json` file at the root of the project.

### File Structure

```json
{
  "defaultSettings": {
    "speed": 1.0,
    "volume": 0,
    "pitch": 1.0,
    "emotion": null,
    "intensity": 0.5,
    "style": null,
    "checkpointPath": "checkpoints/SPEAKER_1-merged/"
  },
  "voices": {
    "VOICE_NAME": {
      "speed": 1.15,
      "volume": 6,
      "pitch": 1.0,
      "emotion": "neutral",
      "intensity": 0.7,
      "style": "casual",
      "checkpointPath": "checkpoints/custom-model/",
      "description": "Voice description"
    }
  }
}
```

## Voice Parameters

### Basic Parameters

- **speed** (0.5-2.0): Speech speed
  - `1.0` = normal speed
  - `1.15` = 15% faster
  - `0.85` = 15% slower

- **volume** (-20 to +20): Volume in decibels
  - `0` = normal volume
  - `+6` = about 20% louder
  - `-6` = about 20% quieter

- **pitch** (0.5-2.0): Pitch
  - `1.0` = normal pitch
  - `1.2` = 20% higher
  - `0.8` = 20% lower

### Model Parameters

- **checkpointPath**: Path to the model checkpoint
  - Can be `null` - then the default path will be used
  - Points to a directory with a fine-tuned model
  - Examples: `"checkpoints/SPEAKER_1-merged/"`, `"checkpoints/custom-voice/"`
  - Path is relative to the `fs-python/` directory

### Emotional Parameters

- **emotion**: Speech emotion
  - `"happy"` - happy
  - `"sad"` - sad
  - `"angry"` - angry
  - `"neutral"` - neutral
  - `"excited"` - excited
  - `null` - no emotion

- **intensity** (0.0-1.0): Emotion intensity
  - `0.0` = minimum intensity
  - `0.5` = medium intensity
  - `1.0` = maximum intensity

- **style**: Speech style
  - `"formal"` - formal
  - `"casual"` - casual
  - `"dramatic"` - dramatic
  - `null` - no style

## Configuration Examples

### Fast energetic voice with a custom model
```json
"FastSpeaker": {
  "speed": 1.3,
  "volume": 3,
  "pitch": 1.1,
  "emotion": "excited",
  "intensity": 0.8,
  "style": "casual",
  "checkpointPath": "checkpoints/energetic-voice/"
}
```

### Slow serious voice with the base model
```json
"SlowSpeaker": {
  "speed": 0.8,
  "volume": -2,
  "pitch": 0.9,
  "emotion": "neutral",
  "intensity": 0.3,
  "style": "formal",
  "checkpointPath": null
}
```

### Loud voice for accents with a fine-tuned model
```json
"LoudSpeaker": {
  "speed": 1.0,
  "volume": 8,
  "pitch": 1.0,
  "emotion": null,
  "intensity": 0.5,
  "style": null,
  "checkpointPath": "checkpoints/narrator-voice-v2/"
}
```

## Setting up fine-tuned models

### Using custom checkpoints

To use your own trained models:

1. Place the checkpoint folder in `fs-python/checkpoints/`
2. Specify the path to this folder in the `checkpointPath` parameter
3. The path should be relative to the `fs-python/` directory

Example structure:
```
fs-python/
├── checkpoints/
│   ├── SPEAKER_1-merged/          # Base model
│   ├── my-custom-voice/           # Your model
│   │   ├── model.pth
│   │   ├── config.json
│   │   └── ...
│   └── narrator-professional/     # Another model
└── ...
```

### Configuration for custom models
```json
"MyCustomVoice": {
  "speed": 1.0,
  "volume": 0,
  "pitch": 1.0,
  "emotion": null,
  "intensity": 0.5,
  "style": null,
  "checkpointPath": "checkpoints/my-custom-voice/",
  "description": "My fine-tuned model"
}
```

## How to add a new voice

1. Open the `voice-config.json` file
2. Add a new entry to the `"voices"` section
3. Specify the voice name and parameters
4. If using a custom model, specify `checkpointPath`
5. Save the file

Example:
```json
"voices": {
  "MyNewVoice": {
    "speed": 1.1,
    "volume": 2,
    "pitch": 1.0,
    "emotion": "happy",
    "intensity": 0.6,
    "style": "casual",
    "checkpointPath": "checkpoints/my-new-model/",
    "description": "My new voice with slight speed-up and custom model"
  }
}
```

## Current settings

The file already contains the following voices:

- **RU_Female_Kropina_YouTube**: 
  - Speed 1.05, volume +6dB (~+20%)
  - Uses the default model (`checkpointPath: null`)
  
- **RU_Male_Goblin_Puchkov**: 
  - Speed 1.05
  - Uses fine-tuned model `checkpoints/SPEAKER_1-merged/`

## Model selection logic

The system selects the model in the following order:

1. **Individual voice setting**: If the voice has `checkpointPath` specified
2. **Default settings**: If the voice has `checkpointPath: null`, the path from `defaultSettings` is used
3. **Fallback**: If `defaultSettings` also has `null`, `"checkpoints/SPEAKER_1-merged/"` is used

## Debugging

If the voice is not found in the configuration, default settings will be used.

Logs show which parameters are applied for each voice:
```
Voice settings: speed=1.15, volume=6dB, pitch=1.0
Using checkpoint: checkpoints/SPEAKER_1-merged/
```

### Possible checkpoint issues

- **Model not found**: Check the correctness of the `checkpointPath`
- **Loading errors**: Make sure all model files are present
- **Invalid format**: Check model compatibility with Fish Speech 