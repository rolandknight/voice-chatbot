"""FastAPI surface for poc-tts.

Never imports chatterbox_flash. The engine arrives as a constructor argument
so tests can inject a mock and run without a GPU.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

from poc_tts.config import load_config, voice_paths as configured_voice_paths
from poc_tts.engine_flash import OutOfMemoryError, discover_voices
from poc_tts.models import FlashTTSRequest

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 [-1, 1] as 16-bit PCM WAV."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _ui_shaped_config(config: dict, ui_state: dict | None = None) -> dict:
    """Reshape config.yaml into the keys ui/script.js's initializeUI expects.

    script.js:699 reads config.audio_output.format, falling back to 'mp3'
    when the key is absent -- and config.yaml never carried an
    audio_output section, so every page load silently selected mp3, which
    models.FlashTTSRequest used to reject outright. The PoC only ever
    encodes and returns WAV, so this is pinned to "wav" unconditionally,
    regardless of whatever config.yaml may say.

    script.js:689 reads config.generation_defaults, with cfg_scale renamed
    to cfg_weight at script.js:694 -- the name FlashTTSRequest.cfg_weight
    uses. Without this mapping the 'generation:' block in config.yaml was
    dead for all GUI traffic; script.js found no generation_defaults and
    fell back to its own hardcoded slider defaults.

    script.js:621 reads config.ui_state and restores last_text and
    last_preset_name from it. Without it every page load saw an empty
    textarea, took the "no text" branch at script.js:711, and re-applied the
    default preset -- so a preset could never be cleared. /save_settings
    already collects that state; this is what hands it back.

    Note the shape: script.js:278 posts {"ui_state": {...}}, so the settings
    store holds that wrapper and the caller must pass the INNER dict. Passing
    the whole store double-nests it and script.js reads undefined.

    The raw config.yaml keys (engine, generation, voices, server, ...) pass
    through unchanged via the spread below, so nothing that already worked
    is lost -- callers that want the PoC's own config shape still have it.
    """
    generation = dict(config.get("generation", {}))
    if "cfg_scale" in generation:
        generation["cfg_weight"] = generation.pop("cfg_scale")
    return {
        **config,
        "audio_output": {**config.get("audio_output", {}), "format": "wav"},
        "generation_defaults": generation,
        "ui_state": dict(ui_state or {}),
    }


# The UI is edited in place during PoC work; a cached script.js silently
# serves stale behaviour and makes browser verification lie. Never cache it.
_NO_STORE = {"Cache-Control": "no-store, must-revalidate"}


def _voice_record(name: str) -> dict:
    """Shape one reference filename the way ui/script.js expects.

    Used by both /get_predefined_voices and /api/ui/initial-data -- they must
    agree, so the derivation lives in one place.
    """
    return {
        "display_name": Path(name).stem.replace("_", " ").title(),
        "filename": name,
    }


def create_app(engine, config: dict, voice_paths: list[Path]) -> FastAPI:
    app = FastAPI(title="poc-tts: Chatterbox Flash", version="0.1.0")

    # Round-tripped to the UI as config.ui_state so preset/text choices stick.
    settings_store: dict = {}

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(UI_DIR / "index.html", headers=_NO_STORE)

    @app.get("/script.js", include_in_schema=False)
    async def script_js():
        return FileResponse(UI_DIR / "script.js", media_type="application/javascript", headers=_NO_STORE)

    @app.get("/styles.css", include_in_schema=False)
    async def styles_css():
        return FileResponse(UI_DIR / "styles.css", media_type="text/css", headers=_NO_STORE)

    @app.get("/vendor/wavesurfer.min.js", include_in_schema=False)
    async def wavesurfer_js():
        return FileResponse(
            UI_DIR / "vendor" / "wavesurfer.min.js", media_type="application/javascript"
        )

    @app.get("/api/model-info")
    async def model_info():
        return engine.model_info()

    @app.get("/get_reference_files")
    async def get_reference_files():
        return discover_voices(voice_paths)

    @app.get("/get_predefined_voices")
    async def get_predefined_voices():
        return [_voice_record(name) for name in discover_voices(voice_paths)]

    @app.get("/api/ui/initial-data")
    async def initial_data():
        presets = []
        presets_file = UI_DIR / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
                if isinstance(loaded, list):
                    presets = loaded
        names = discover_voices(voice_paths)
        return {
            "config": _ui_shaped_config(config, settings_store.get("ui_state", {})),
            "reference_files": names,
            "predefined_voices": [_voice_record(n) for n in names],
            "presets": presets,
            "initial_gen_result": {
                "outputUrl": None, "filename": None, "genTime": None,
                "submittedVoiceMode": None, "submittedPredefinedVoice": None,
                "submittedCloneFile": None,
            },
            "model_info": engine.model_info(),
        }

    @app.post("/restart_server")
    async def restart_server():
        return JSONResponse(
            {"message": "Restarting is not supported in the poc-tts PoC. "
                        "Stop and rerun `make poc-tts`."}
        )


    @app.post("/tts")
    async def tts(request: FlashTTSRequest):
        if not engine.loaded:
            raise HTTPException(status_code=503, detail="Flash model is not loaded.")

        if request.voice_mode == "predefined":
            voice = request.predefined_voice_id
            if not voice:
                raise HTTPException(
                    status_code=400,
                    detail="predefined_voice_id is required when voice_mode is 'predefined'.",
                )
        else:
            voice = request.reference_audio_filename
            if not voice:
                raise HTTPException(
                    status_code=400,
                    detail="reference_audio_filename is required when voice_mode is 'clone'.",
                )

        try:
            audio, sample_rate = engine.synthesize(
                text=request.text,
                voice=voice,
                temperature=request.temperature,
                exaggeration=request.exaggeration,
                cfg_scale=request.cfg_weight,
                num_steps=request.num_steps,
                n_cfm_timesteps=request.n_cfm_timesteps,
                chunk_size=request.chunk_size,
                split_text=request.split_text,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OutOfMemoryError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(content=_wav_bytes(audio, sample_rate), media_type="audio/wav")

    @app.post("/save_settings")
    async def save_settings(payload: dict):
        settings_store.update(payload)
        return {"message": "Settings saved."}

    @app.post("/reset_settings")
    async def reset_settings():
        settings_store.clear()
        return {"message": "Settings reset."}

    return app


def main() -> None:
    import sys

    import uvicorn

    from poc_tts.engine_flash import FlashEngine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    paths = configured_voice_paths(config)
    engine = FlashEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=config.get("generation", {}),
        voice_paths=paths,
    )
    try:
        engine.load()
    except OutOfMemoryError as exc:
        logger.error("failed to load Chatterbox Flash: %s", exc)
        sys.exit(f"poc-tts: failed to load the model -- {exc}")
    app = create_app(engine, config, voice_paths=paths)
    uvicorn.run(
        app,
        host=config.get("server", {}).get("host", "127.0.0.1"),
        port=config.get("server", {}).get("port", 8005),
    )


if __name__ == "__main__":
    main()
