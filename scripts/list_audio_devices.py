#!/usr/bin/env python3
import pyaudio

pa = pyaudio.PyAudio()
print("\nPyAudio devices:\n")
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    name = d.get("name", "")
    max_in = int(d.get("maxInputChannels", 0))
    max_out = int(d.get("maxOutputChannels", 0))
    default_sr = int(float(d.get("defaultSampleRate", 0)))
    marker = "  <== likely Jabra" if "jabra" in name.lower() else ""
    print(f"[{i:2d}] in={max_in:2d} out={max_out:2d} sr={default_sr:5d}  {name}{marker}")

try:
    print("\nDefault input :", pa.get_default_input_device_info()["name"])
except Exception as e:
    print("\nDefault input : unavailable", e)

try:
    print("Default output:", pa.get_default_output_device_info()["name"])
except Exception as e:
    print("Default output: unavailable", e)

pa.terminate()

print("""
Prefer NAMES over indexes — PyAudio indexes are not stable across runs on macOS.
Put a unique substring of the device name in .env, for example:

INPUT_DEVICE_NAME=Jabra
OUTPUT_DEVICE_NAME=Jabra

For a USB speakerphone, the same name often matches both input and output entries.
""")
