"""Gradio app mirroring https://huggingface.co/spaces/Qwen/Qwen3-TTS.

Three tabs: Voice Design, Voice Clone, TTS (CustomVoice). Never imports
mlx_audio; everything goes through Qwen3Engine. Every handler returns
(audio, status) and turns exceptions into a status message.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .config import POC_DIR, load_config, voice_dirs
from .engine import LANGUAGES, SIZES, Qwen3Engine, discover_voices, sidecar_transcript

log = logging.getLogger(__name__)
UI_RUNS = POC_DIR / "reports" / "ui_runs.jsonl"

EXAMPLES = {
    "design_text": "It's in the top drawer... wait, it's empty?",
    "design_instruct": "A cheerful young female voice with high pitch and energetic tone.",
    "clone_text": "I checked the calendar for tomorrow and you have three meetings, the first one starting at nine fifteen.",
    "tts_text": "Sure, the kitchen light is on.",
    "tts_instruct": "Very happy and excited.",
}


def status_line(tab: str, timings: dict) -> str:
    return (
        f"**{tab}** · `{timings['model'].split('/')[-1]}` · {timings['chars']} chars → "
        f"{timings['audio_s']:.2f} s audio in **{timings['gen_s']:.2f} s** (RTF {timings['rtf']:.2f}, "
        f"{timings['chunks']} chunk{'s' if timings['chunks'] != 1 else ''})"
    )


def record_run(tab: str, timings: dict) -> None:
    try:
        UI_RUNS.parent.mkdir(parents=True, exist_ok=True)
        with open(UI_RUNS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "tab": tab, **timings}) + "\n")
    except OSError:
        pass


def _audio_out(result):
    return (result.sample_rate, result.audio)


class Handlers:
    """Plain callables (testable without Gradio) that the Blocks wires to buttons."""

    def __init__(self, engine: Qwen3Engine, voices: dict[str, Path]):
        self.engine = engine
        self.voices = voices

    def _guard(self, tab, fn):
        try:
            result = fn()
        except Exception as exc:  # surfaced in the status box, never crashes the server
            log.error("%s failed: %s\n%s", tab, exc, traceback.format_exc())
            return None, f"❌ **{tab} failed:** {exc}"
        record_run(tab, result.timings)
        return _audio_out(result), status_line(tab, result.timings)

    def voice_design(self, text, language, instruct):
        return self._guard("Voice Design", lambda: self.engine.voice_design(text, instruct, language=language))

    def voice_clone(self, ref_audio, ref_text, xvector_only, text, language, size):
        if ref_audio is None:
            return None, "❌ Upload or record a reference clip, or pick a preset voice."
        return self._guard(
            "Voice Clone",
            lambda: self.engine.clone(text, ref_audio, ref_text, language=language, size=size, xvector_only=bool(xvector_only)),
        )

    def custom_voice(self, text, language, speaker, instruct, size):
        return self._guard("TTS", lambda: self.engine.custom_voice(text, speaker, language=language, instruct=instruct, size=size))

    def pick_preset(self, name):
        """Preset dropdown -> (audio path, transcript from sidecar or whisper)."""
        if not name or name not in self.voices:
            return None, ""
        clip = self.voices[name]
        text = sidecar_transcript(clip)
        if text is None:
            text = self.engine.transcribe(clip)
        return str(clip), text

    def transcribe(self, ref_audio):
        if ref_audio is None:
            return ""
        return self.engine.transcribe(ref_audio)

    def unload(self):
        self.engine.unload_all()
        return self.info_md()

    def info_md(self):
        i = self.engine.model_info()
        resident = ", ".join(m.split("/")[-1] for m in i.get("resident", [])) or "none"
        return (
            f"{i.get('chip', '?')} · mlx {i.get('mlx', '?')} · mlx-audio {i.get('mlx_audio', '?')} · "
            f"resident: {resident} · active {i.get('active_gb', '?')} GiB · peak {i.get('peak_gb', '?')} GiB"
        )


def build_demo(handlers: Handlers):
    import gradio as gr

    langs = list(LANGUAGES)
    with gr.Blocks(title="Qwen3-TTS — poc-qwen") as demo:
        gr.Markdown("# Qwen3-TTS on Apple Silicon (mlx-audio)")
        with gr.Row():
            info = gr.Markdown(handlers.info_md())
            unload_btn = gr.Button("Unload models", size="sm", scale=0)
        unload_btn.click(handlers.unload, outputs=info)

        with gr.Tab("Voice Design"):
            d_text = gr.Textbox(label="Text to Synthesize", value=EXAMPLES["design_text"], lines=3)
            d_lang = gr.Dropdown(label="Language", choices=langs, value="Auto")
            d_instruct = gr.Textbox(label="Voice Description", value=EXAMPLES["design_instruct"], lines=2)
            d_btn = gr.Button("Generate", variant="primary")
            d_audio = gr.Audio(label="Generated Audio", type="numpy")
            d_status = gr.Markdown()
            d_btn.click(handlers.voice_design, [d_text, d_lang, d_instruct], [d_audio, d_status]).then(handlers.info_md, outputs=info)

        with gr.Tab("Voice Clone"):
            with gr.Row():
                with gr.Column():
                    c_preset = gr.Dropdown(label="Preset voice", choices=[""] + list(handlers.voices), value="")
                    c_ref = gr.Audio(label="Reference Audio", sources=["upload", "microphone"], type="filepath")
                    c_ref_text = gr.Textbox(label="Reference Text", lines=3, placeholder="Transcript of the reference clip (required unless x-vector only)")
                    with gr.Row():
                        c_transcribe = gr.Button("Auto-transcribe", size="sm")
                        c_xvec = gr.Checkbox(label="Use x-vector only", value=False)
                with gr.Column():
                    c_text = gr.Textbox(label="Target Text", value=EXAMPLES["clone_text"], lines=4)
                    c_lang = gr.Dropdown(label="Language", choices=langs, value="Auto")
                    c_size = gr.Radio(label="Model Size", choices=list(SIZES), value="1.7B")
                    c_btn = gr.Button("Generate", variant="primary")
            c_audio = gr.Audio(label="Generated Audio", type="numpy")
            c_status = gr.Markdown()
            c_preset.change(handlers.pick_preset, c_preset, [c_ref, c_ref_text])
            c_transcribe.click(handlers.transcribe, c_ref, c_ref_text)
            c_btn.click(handlers.voice_clone, [c_ref, c_ref_text, c_xvec, c_text, c_lang, c_size], [c_audio, c_status]).then(handlers.info_md, outputs=info)

        with gr.Tab("TTS (CustomVoice)"):
            t_text = gr.Textbox(label="Text to Synthesize", value=EXAMPLES["tts_text"], lines=3)
            t_lang = gr.Dropdown(label="Language", choices=langs, value="Auto")
            t_speaker = gr.Dropdown(label="Speaker", choices=handlers.engine.speakers(), value=handlers.engine.speakers()[0])
            t_instruct = gr.Textbox(label="Style Instruction (Optional)", placeholder=EXAMPLES["tts_instruct"])
            t_size = gr.Dropdown(label="Model Size", choices=list(SIZES), value="1.7B")
            t_btn = gr.Button("Generate", variant="primary")
            t_audio = gr.Audio(label="Generated Audio", type="numpy")
            t_status = gr.Markdown()
            t_btn.click(handlers.custom_voice, [t_text, t_lang, t_speaker, t_instruct, t_size], [t_audio, t_status]).then(handlers.info_md, outputs=info)
    return demo


def main() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    engine = Qwen3Engine(cfg)
    voices = discover_voices(voice_dirs(cfg))
    log.info("voices: %s", list(voices))
    demo = build_demo(Handlers(engine, voices))
    host = os.environ.get("HOST", cfg["server"]["host"])
    demo.queue(default_concurrency_limit=1).launch(server_name=host, server_port=int(cfg["server"]["port"]), share=False, show_error=True)


if __name__ == "__main__":
    main()
