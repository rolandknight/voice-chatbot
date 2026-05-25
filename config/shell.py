"""Emit a fixed allowlist of config values as shell-quoted KEY=VALUE lines.

run.sh evals the output:  eval "$(.venv/bin/python -m config.shell)"

Only values bash itself consumes (sidecar launch gating) are listed here.
"""

from __future__ import annotations

import shlex

from config.loader import load


def main() -> None:
    cfg = load()
    pairs = [
        ("PERSONAS_CONFIG", str(cfg.resolved_personas_path())),
        ("CHATTERBOX_BASE_URL", cfg.tts.chatterbox.base_url),
        ("BABEL_SFX_ENABLED", "1" if cfg.skills.sfx.woosh_enabled else "0"),
        ("WOOSH_PORT", str(cfg.skills.sfx.woosh_port)),
        ("WOOSH_URL", cfg.skills.sfx.woosh_url),
        ("BABEL_SAO_ENABLED", "1" if cfg.skills.sfx.sao_enabled else "0"),
        ("STABLE_AUDIO_PORT", str(cfg.skills.sfx.sao_port)),
        ("STABLE_AUDIO_URL", cfg.skills.sfx.sao_url),
        ("OLLAMA_KEEP_ALIVE", cfg.llm.ollama_keep_alive),
    ]
    for key, value in pairs:
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
