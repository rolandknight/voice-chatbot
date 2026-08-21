#!/usr/bin/env bash
# Clone (or locate) Chatterbox-TTS-Server without replacing an existing venv.
# Dependency installation and model download happen on first start.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_chatterbox_common.sh
. "$PROJECT_DIR/scripts/_chatterbox_common.sh"

REPO_URL="https://github.com/devnen/Chatterbox-TTS-Server.git"
SERVER_DIR="$(chatterbox_resolve_server_dir "$PROJECT_DIR")"

mkdir -p "$(dirname "$SERVER_DIR")"

if [ ! -d "$SERVER_DIR/.git" ]; then
    echo "Cloning Chatterbox-TTS-Server into $SERVER_DIR ..."
    git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
elif [ "${CHATTERBOX_UPDATE:-0}" = "1" ]; then
    echo "Updating Chatterbox-TTS-Server at $SERVER_DIR ..."
    git -C "$SERVER_DIR" pull --ff-only || {
        echo "Update skipped; the checkout is not fast-forwardable." >&2
    }
else
    echo "Chatterbox-TTS-Server already present at $SERVER_DIR."
    echo "Set CHATTERBOX_UPDATE=1 to request a fast-forward update."
fi

echo ""
echo "Repo ready at $SERVER_DIR."
echo "Next: ./scripts/start_chatterbox.sh"
echo "The first launch installs dependencies and downloads the model (several GB)."
