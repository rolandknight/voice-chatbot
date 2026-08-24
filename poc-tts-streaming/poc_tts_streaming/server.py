"""FastAPI surface for poc-tts.

Never imports chatterbox_flash. The engine arrives as a constructor argument
so tests can inject a mock and run without a GPU.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Callable

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from poc_tts_streaming.config import load_config, voice_paths as configured_voice_paths
from poc_tts_streaming.engine_flash import OutOfMemoryError, discover_voices
from poc_tts_streaming.realtime import ids
from poc_tts_streaming.realtime.events import EventError
from poc_tts_streaming.realtime.session import ChatterboxKnobs, RealtimeSession, SynthWorker

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def openai_error(status: int, message: str, *, type_: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None) -> JSONResponse:
    """The error body shape api.openai.com returns, so client code paths match."""
    return JSONResponse({"error": {"type": type_, "code": code, "message": message, "param": param}},
                        status_code=status)


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


class ClientSecretStore:
    """In-memory ephemeral keys. Cosmetic on localhost, but it keeps the
    browser's code path identical to the one it would use against OpenAI."""

    def __init__(self, ttl_s: int = 600, clock: Callable[[], int] = ids.now) -> None:
        self._ttl, self._clock = ttl_s, clock
        self._tokens: dict[str, int] = {}

    def issue(self, session_patch: dict | None, *, session_factory=None) -> dict:
        session = session_factory(session_patch).session_object() if session_factory else {}
        value = f"ek_{secrets.token_urlsafe(24)}"
        expires_at = self._clock() + self._ttl
        self._tokens[value] = expires_at
        return {"value": value, "expires_at": expires_at, "session": session}

    def verify(self, token: str | None) -> bool:
        if not token or token not in self._tokens:
            return False
        if self._tokens[token] < self._clock():
            del self._tokens[token]
            return False
        return True


def engine_synthesizer(engine):
    """Adapt FlashEngine.synthesize_stream to the session's Synthesizer type."""
    def synthesize(text, voice, knobs: ChatterboxKnobs, cancel):
        return engine.synthesize_stream(text, voice, cancel=cancel, **knobs.as_engine_kwargs())
    return synthesize


class _NullSink:
    def push(self, pcm): ...
    def flush(self): ...
    def clear(self): ...
    async def drained(self): ...


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


def create_app(engine, config: dict, voice_paths: list[Path], *,
               worker: SynthWorker | None = None) -> FastAPI:
    app = FastAPI(
        title="poc-tts-streaming: Chatterbox Flash over Realtime/WebRTC", version="0.1.0"
    )
    realtime_cfg = {"model": "chatterbox-flash", "default_voice": "one-one.mp3",
                    "client_secret_ttl_s": 600, **config.get("realtime", {})}
    app.state.knobs = ChatterboxKnobs.from_config(config.get("generation", {}))
    app.state.worker = worker or SynthWorker()
    app.state.secrets = ClientSecretStore(ttl_s=int(realtime_cfg["client_secret_ttl_s"]))
    app.state.realtime = realtime_cfg

    def build_session(send, sink, session_patch: dict | None = None) -> RealtimeSession:
        return RealtimeSession(
            send=send, synthesizer=engine_synthesizer(engine), sink=sink, worker=app.state.worker,
            voices=lambda: discover_voices(voice_paths), voice=realtime_cfg["default_voice"],
            knobs=app.state.knobs, model=realtime_cfg["model"], session_patch=session_patch,
        )
    app.state.build_session = build_session

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

    @app.post("/v1/realtime/client_secrets")
    async def client_secrets(request: Request):
        body = await request.json() if int(request.headers.get("content-length", "0") or 0) else {}
        patch = body.get("session") if isinstance(body, dict) else None
        try:
            return app.state.secrets.issue(
                patch, session_factory=lambda p: build_session(lambda _e: None, _NullSink(), p))
        except EventError as err:
            return openai_error(400, err.message, code=err.code, param=err.param)

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

    from poc_tts_streaming.engine_flash import FlashEngine

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
        port=config.get("server", {}).get("port", 8006),
    )


if __name__ == "__main__":
    main()
