#!/usr/bin/env bash
# Install the Raspberry Pi 5 wake-client dependencies.
#
# Works on Python 3.11 AND 3.12. openWakeWord 0.6.0 declares a Linux-only
# dependency on `tflite-runtime`, which has no wheels for Python >= 3.12, so a
# plain `pip install openwakeword` fails there with "requires Python < 3.12".
# We use the ONNX inference path (the tflite import is lazy and unused), so we
# install openWakeWord with --no-deps and provide its real runtime deps via
# requirements.txt.
#
# Usage:  ./install_rpi.sh            (installs into the active python env; run
#                                      `. bin/activate-hermit` from the repo root first)
#         PYTHON=python3.12 ./install_rpi.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

# sounddevice binds to the system PortAudio library — it is NOT a pip dep.
if ! ldconfig -p 2>/dev/null | grep -q "libportaudio\.so\.2"; then
  echo "ERROR: PortAudio not found. Install it first:" >&2
  echo "  sudo apt install -y libportaudio2 portaudio19-dev" >&2
  exit 1
fi

"$PY" -m pip install --upgrade pip wheel
"$PY" -m pip install -r "$HERE/requirements.txt"
# --no-deps: skip openWakeWord's Linux-only tflite-runtime pin (unused; we run ONNX).
"$PY" -m pip install --no-deps "openwakeword==0.6.0"

# openWakeWord's melspectrogram + embedding backbone (and the Silero VAD) ship
# via CDN, not the wheel. The client loads them at startup to run ANY wake
# model — without them it fails with "cannot find melspectrogram.onnx". Our
# wake models are the custom ones under models/wakeword/, so we only need the
# shared backbone; download_models() always fetches it regardless of the model
# list. Mirrors the server's install_mac.sh.
echo
echo "Downloading openWakeWord shared backbone (melspectrogram/embedding/VAD)..."
"$PY" -c "import openwakeword.utils as u; u.download_models(['hey_jarvis_v0.1'])"

echo
echo "Verifying imports:"
"$PY" -c "import sounddevice; print('  sounddevice + PortAudio OK')"
"$PY" -c "from openwakeword.model import Model; print('  openWakeWord OK (onnx)')"
# Fail loudly at install time if the backbone is still missing, rather than at
# first wake on the device.
"$PY" - <<'EOF'
import os, openwakeword
res = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
missing = [f for f in ("melspectrogram.onnx", "embedding_model.onnx") if not os.path.exists(os.path.join(res, f))]
if missing:
    raise SystemExit(f"ERROR: openWakeWord backbone missing after download: {missing} (in {res})")
print("  openWakeWord backbone OK (melspectrogram + embedding)")
EOF
echo "Done."
