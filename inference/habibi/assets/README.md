# Habibi TTS — Reference Audio Assets

Place your reference audio files here to define custom voices.

## Format

For each voice, add two files:
- `<voice_name>.wav`  — 6–15 seconds of clean Arabic speech, 24kHz mono
- `<voice_name>.txt`  — exact Arabic transcript of the wav file

## Example

```
assets/
  fahad.wav
  fahad.txt
  layla.wav
  layla.txt
```

## Setting the active voice

Set `HABIBI_VOICE=fahad` (or any name) in your `.env` file.
The server will load the matching wav+txt pair from this directory.

## Fallback

If the requested voice file is not found here, the server falls back
to the built-in default voice bundled with habibi-tts.

## Tips

- Cleaner reference audio → better output quality
- 8–12 seconds is the sweet spot
- Avoid background noise or music in the reference
- The transcript must match the audio exactly (no punctuation differences)
