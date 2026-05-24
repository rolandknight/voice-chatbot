#!/usr/bin/env python3
"""Confirm Pipecat's LocalAudioTransport actually delivers audio frames
and that Silero VAD sees speech. Runs for ~10 seconds. Speak during the run.
"""
import asyncio
import os
import sys
import time

import audioop
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from _audio_devices import resolve_from_env  # noqa: E402

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


class AudioInspector(FrameProcessor):
    def __init__(self, vad: SileroVADAnalyzer):
        super().__init__()
        self._vad = vad
        self._sr_set = False
        self._chunks = 0
        self._peak_rms = 0
        self._max_conf = 0.0
        self._last_print = time.time()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            if not self._sr_set:
                self._vad.set_sample_rate(frame.sample_rate)
                self._sr_set = True
            self._chunks += 1
            rms = audioop.rms(frame.audio, 2)
            self._peak_rms = max(self._peak_rms, rms)
            try:
                conf = self._vad.voice_confidence(frame.audio)
                self._max_conf = max(self._max_conf, conf)
            except Exception:
                pass
            now = time.time()
            if now - self._last_print >= 1.0:
                print(
                    f"  audio_frames={self._chunks}  "
                    f"peak_rms={self._peak_rms}  "
                    f"max_silero_conf={self._max_conf:.3f}"
                )
                self._chunks = 0
                self._peak_rms = 0
                self._max_conf = 0.0
                self._last_print = now
        await self.push_frame(frame, direction)


async def main():
    load_dotenv(override=True)
    in_idx, out_idx, in_name, out_name = resolve_from_env()
    print(f"Input  : [{in_idx}] {in_name}")
    print(f"Output : [{out_idx}] {out_name}")

    sr = int(os.getenv("AUDIO_IN_SAMPLE_RATE", "16000"))
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            audio_in_sample_rate=sr,
            audio_in_channels=1,
            audio_in_passthrough=True,
            input_device_index=in_idx,
        )
    )

    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.0))
    inspector = AudioInspector(vad)
    task = PipelineTask(Pipeline([transport.input(), inspector]), params=PipelineParams())

    async def stopper():
        await asyncio.sleep(10)
        await task.cancel()

    print("Speak now (10s). Report once per second:\n")
    await asyncio.gather(PipelineRunner().run(task), stopper())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
