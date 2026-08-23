"""FastAPI surface for poc-tts.

Never imports chatterbox_flash. The engine arrives as a constructor argument
so tests can inject a mock and run without a GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from poc_tts.config import load_config, voice_paths as configured_voice_paths
from poc_tts.engine_flash import discover_voices

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _voice_record(name: str) -> dict:
    """Shape one reference filename the way ui/script.js expects.

    Used by both /get_predefined_voices and /api/ui/initial-data -- they must
    agree, so the derivation lives in one place.
    """
    return {
        "display_name": name.replace(".wav", "").replace("_", " ").title(),
        "filename": name,
    }


def create_app(engine, config: dict, voice_paths: list[Path]) -> FastAPI:
    app = FastAPI(title="poc-tts: Chatterbox Flash", version="0.1.0")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/script.js", include_in_schema=False)
    async def script_js():
        return FileResponse(UI_DIR / "script.js", media_type="application/javascript")

    @app.get("/styles.css", include_in_schema=False)
    async def styles_css():
        return FileResponse(UI_DIR / "styles.css", media_type="text/css")

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
            "config": config,
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

    return app


def main() -> None:
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
    engine.load()
    app = create_app(engine, config, voice_paths=paths)
    uvicorn.run(
        app,
        host=config.get("server", {}).get("host", "127.0.0.1"),
        port=config.get("server", {}).get("port", 8005),
    )


if __name__ == "__main__":
    main()
