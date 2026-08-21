#!/usr/bin/env bash
# Shared Chatterbox path/environment discovery for macOS and Linux launchers.
# This file is sourced; callers remain responsible for `set -euo pipefail`.

chatterbox_python_is_usable() {
    local python_path="$1"
    [ -x "$python_path" ] || return 1
    "$python_path" -c 'import torch, uvicorn, chatterbox' >/dev/null 2>&1
}

chatterbox_find_python() {
    local server_dir="$1"
    local candidate

    # .venv was used by older installs in this repository; current upstream
    # uses venv. Accept both and never delete either automatically.
    for candidate in \
        "$server_dir/.venv/bin/python" \
        "$server_dir/venv/bin/python"; do
        if chatterbox_python_is_usable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

chatterbox_resolve_server_dir() {
    local project_dir="$1"
    local canonical="$project_dir/vendor/chatterbox-tts-server"
    local legacy="$project_dir/vendor/Chatterbox-TTS-Server"
    local candidate

    if [ -n "${CHATTERBOX_SERVER_DIR:-}" ]; then
        case "$CHATTERBOX_SERVER_DIR" in
        /*) candidate="$CHATTERBOX_SERVER_DIR" ;;
        *) candidate="$project_dir/$CHATTERBOX_SERVER_DIR" ;;
        esac
        printf '%s\n' "$candidate"
        return 0
    fi

    # Prefer a clone that already has a usable environment. This avoids a
    # second multi-gigabyte install on hosts with the older capitalized path.
    for candidate in "$canonical" "$legacy"; do
        if chatterbox_find_python "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    # Otherwise reuse either clone before choosing the canonical new path.
    for candidate in "$canonical" "$legacy"; do
        if [ -d "$candidate/.git" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    printf '%s\n' "$canonical"
}

chatterbox_health_url() {
    local base_url="${1%/}"
    case "$base_url" in
    */v1) printf '%s/audio/voices\n' "$base_url" ;;
    *) printf '%s/v1/audio/voices\n' "$base_url" ;;
    esac
}
