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
# Usage:  ./install_rpi.sh            (uses ./.venv or the active venv)
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

echo
echo "Verifying imports:"
"$PY" -c "import sounddevice; print('  sounddevice + PortAudio OK')"
"$PY" -c "from openwakeword.model import Model; print('  openWakeWord OK (onnx)')"
echo "Done."
