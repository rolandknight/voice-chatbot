#!/usr/bin/env python3
"""Capture 3 seconds from the resolved input device and report level."""
import os
import sys
import time
import audioop

import pyaudio
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from _audio_devices import resolve_from_env  # noqa: E402

load_dotenv()
sr = int(os.getenv("AUDIO_IN_SAMPLE_RATE", "16000"))

in_idx, _out_idx, in_name, _out_name = resolve_from_env()

pa = pyaudio.PyAudio()
print(f"Opening input device [{in_idx}] {in_name} @ {sr} Hz")

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=sr,
    input=True,
    input_device_index=in_idx,
    frames_per_buffer=1024,
)

print("Speak now (3s)...")
peak = 0
total = 0
chunks = 0
deadline = time.time() + 3.0
while time.time() < deadline:
    data = stream.read(1024, exception_on_overflow=False)
    rms = audioop.rms(data, 2)
    peak = max(peak, rms)
    total += rms
    chunks += 1

stream.stop_stream()
stream.close()
pa.terminate()

avg = total / max(chunks, 1)
print(f"chunks={chunks}  avg_rms={avg:.1f}  peak_rms={peak}")
if peak < 50:
    print("NEAR-SILENCE. Likely macOS mic permission denied or wrong device.")
    print("Fix: System Settings > Privacy & Security > Microphone — enable for your terminal app.")
    sys.exit(1)
else:
    print("Mic is delivering audio.")
