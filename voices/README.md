# Reference voice clips for Chatterbox personas

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
