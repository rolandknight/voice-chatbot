# Reference voice clips for Chatterbox personas

## Current clips

- `babel.mp3` — the `babel` persona (wake word "hey babel").
- `marvin.mp3` — the `marvin` persona (wake word "hey marvin").
- `one-one.mp3` — the default/bench reference clip used by
  `poc-tts-streaming`'s `bench.voice` and `realtime.default_voice`.

This is a curated list, deliberately kept short; poc-tts-streaming's
`config.yaml` only searches this directory (`voices.paths: [../voices]`) so
whatever lands here becomes the full predefined-voice list there.

Drop a short reference clip per cloned persona here. Recommended:

- **5–15 seconds**
- **Mono**
- **24 kHz** preferred (Chatterbox resamples if needed)
- **WAV** preferred (MP3/FLAC also work)
- **Clean** — no music, minimal room reverb, single speaker

Once a clip is in place, add a persona entry to `../personas.yaml`:

```yaml
personas:
  jeeves:
    backend: chatterbox
    voice: jeeves
    ref_audio: voices/jeeves.wav
```

The Chatterbox-TTS-Server picks up the file at startup; restart the
server (Ctrl+C, then `./scripts/start_chatterbox.sh`) after adding new
voices, or run `python scripts/chatterbox_health.py reload`.
