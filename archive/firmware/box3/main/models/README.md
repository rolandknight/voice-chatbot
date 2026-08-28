# microWakeWord models

The firmware embeds two `.tflite` files at build time via `EMBED_FILES`:

- `hey_babel.tflite`  — routes to the Ollama backend
- `hey_marvin.tflite` — routes to the Claude backend

These files are **not in git** (each is ~50 KB but they regenerate from `scripts/microwakeword/`).

## Get the models

Either train them and install via the firmware Makefile:

```sh
(cd ../../../../scripts/microwakeword && make all)   # train
(cd ../../ && make install)                          # copy into main/models/
```

Or drop in a stock microWakeWord model to validate the firmware first:

```sh
(cd ../../ && make install-stock)   # downloads okay_nabu for both slots
```

The build (`make build` / `idf.py build`) will fail with a missing-file error from `EMBED_FILES` until both files exist — `make build` runs a `check-models` preflight that prints exactly which target to invoke.
