# Voice Testing

## Available Voices

To list all available voices in fs-python:

```bash
cd fs-python
poetry run python cli_tts.py --list-voices
```

## Current Podcast Voices

The following voices are currently used in `conversation.json`:

- **RU_Female_Kropina_YouTube** - female voice (host)
- **RU_Male_Goblin_Puchkov** - male voice (host)

## Quick Voice Testing

```bash
cd fs-python

# Test female voice
poetry run python cli_tts.py "Welcome to our podcast" \
  --voice RU_Female_Kropina_YouTube --play

# Test male voice
poetry run python cli_tts.py "Ready to discuss this topic" \
  --voice RU_Male_Goblin_Puchkov --play
```

## Popular Alternatives

### Female voices:
- `RU_Female_YandexAlisa`
- `RU_Female_AliExpress`
- `RU_Female_IngridOlerinskaya_Judy`

### Male voices:
- `RU_Male_Deadpool`
- `RU_Male_Nagiev`
- `RU_Male_Craster_YouTube`
- `RU_Male_Denis_Kolesnikov`

## Changing Podcast Voices

Edit `conversation.json`:

```json
{
  "conversation": [
    {
      "id": 1,
      "speaker": "NEW_FEMALE_VOICE",
      "text": "..."
    },
    {
      "id": 2,
      "speaker": "NEW_MALE_VOICE", 
      "text": "..."
    }
  ]
}
```

Then run: `npm run process`
