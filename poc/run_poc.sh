#!/usr/bin/env bash
# PoC orchestration (poc/CONTRACT.md "Run orchestration"):
#   run_poc.sh up    - start stubs + selected TTS dependencies + FlowCat
#   run_poc.sh test  - run the smoke suite (pytest -m smoke)
#   run_poc.sh down  - stop everything started by `up`
# Logs: poc/logs/<name>.log   PIDs: poc/logs/<name>.pid
set -euo pipefail

POC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$POC_DIR/.." && pwd)"
VENV="$POC_DIR/.venv"
LOGS="$POC_DIR/logs"
FLOWCAT_BIN="$POC_DIR/flowcat/target/debug/flowcat-poc"

[ -f "$POC_DIR/.env" ] && set -a && . "$POC_DIR/.env" && set +a
TTS_BACKEND="${POC_TTS_BACKEND:-kokoro}"
STT_BACKEND="${POC_STT_BACKEND:-whisper}"

start_proc() { # name workdir cmd...
    local name="$1" workdir="$2"; shift 2
    if [ -f "$LOGS/$name.pid" ] && kill -0 "$(cat "$LOGS/$name.pid")" 2>/dev/null; then
        echo "$name already running (pid $(cat "$LOGS/$name.pid"))"
        return
    fi
    (
        cd "$workdir"
        "$@" >"$LOGS/$name.log" 2>&1 &
        echo $! >"$LOGS/$name.pid"
    )
    echo "$name started (pid $(cat "$LOGS/$name.pid")), log: $LOGS/$name.log"
}

wait_health() { # name url timeout_s
    local name="$1" url="$2" timeout="${3:-30}" i=0
    while ! curl -fsS -o /dev/null "$url" 2>/dev/null; do
        i=$((i + 1))
        if [ "$i" -ge $((timeout * 2)) ]; then
            echo "ERROR: $name not healthy after ${timeout}s ($url) — see $LOGS/$name.log" >&2
            return 1
        fi
        sleep 0.5
    done
    echo "$name healthy ($url)"
}

check_chatterbox() {
    local base_url="${POC_CHATTERBOX_URL:-http://127.0.0.1:8004}"
    local voice="${POC_CHATTERBOX_VOICE:-marvin.wav}"
    local speech_url payload output_file

    base_url="${base_url%/}"
    case "$base_url" in
    */v1) speech_url="$base_url/audio/speech" ;;
    *) speech_url="$base_url/v1/audio/speech" ;;
    esac
    payload="$("$VENV/bin/python" -c \
        'import json,sys; print(json.dumps({"model":"chatterbox","input":"Ready.","voice":sys.argv[1],"response_format":"wav"}))' \
        "$voice")"
    output_file="$LOGS/chatterbox-health.wav"

    if ! curl -fsS --connect-timeout 2 --max-time 180 \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        -o "$output_file" \
        "$speech_url"; then
        echo "ERROR: Chatterbox is not ready at $speech_url." >&2
        echo "Start it in another terminal with: make poc-chatterbox" >&2
        return 1
    fi
    if [ ! -s "$output_file" ] || [ "$(wc -c <"$output_file")" -lt 1000 ]; then
        echo "ERROR: Chatterbox returned invalid audio for voice '$voice'." >&2
        return 1
    fi
    echo "chatterbox healthy ($speech_url, voice=$voice)"
}

ensure_nemotron() {
    local base_url="${POC_NEMOTRON_URL:-http://127.0.0.1:8178}"
    local ready_url="${base_url%/}/ready"

    if curl -fsS --connect-timeout 1 --max-time 2 -o /dev/null "$ready_url" 2>/dev/null; then
        echo "nemotron already healthy ($ready_url)"
        return
    fi

    start_proc nemotron "$REPO_DIR" "$REPO_DIR/scripts/start_nemotron.sh"
    wait_health nemotron "$ready_url" 600
}

up() {
    mkdir -p "$LOGS"
    case "$STT_BACKEND" in
    whisper | moonshine) ;;
    nemotron | nvidia) ensure_nemotron ;;
    *)
        echo "ERROR: unsupported POC_STT_BACKEND '$STT_BACKEND' (expected whisper, moonshine, or nemotron)" >&2
        return 1
        ;;
    esac
    case "$TTS_BACKEND" in
    kokoro) ;;
    chatterbox) check_chatterbox ;;
    *)
        echo "ERROR: unsupported POC_TTS_BACKEND '$TTS_BACKEND' (expected kokoro or chatterbox)" >&2
        return 1
        ;;
    esac
    start_proc stubs "$POC_DIR/stubs" "$VENV/bin/uvicorn" stub_server:app --host 127.0.0.1 --port 8790
    if [ "$TTS_BACKEND" = "kokoro" ]; then
        start_proc kokoro "$POC_DIR/stubs" "$VENV/bin/uvicorn" kokoro_shim:app --host 127.0.0.1 --port 8880
    fi
    if [ -x "$FLOWCAT_BIN" ]; then
        start_proc flowcat "$POC_DIR/flowcat" "$FLOWCAT_BIN"
    else
        echo "NOTE: $FLOWCAT_BIN not built yet — start the FlowCat server separately (Rust side)."
    fi
    wait_health stubs http://127.0.0.1:8790/health 30
    if [ "$TTS_BACKEND" = "kokoro" ]; then
        wait_health kokoro http://127.0.0.1:8880/health 600 # first start downloads ~350 MB of models
    fi
    if [ -x "$FLOWCAT_BIN" ]; then
        wait_health flowcat http://127.0.0.1:6210/healthz 60
    fi
}

down() {
    local pids=()
    for pidfile in "$LOGS"/*.pid; do
        [ -f "$pidfile" ] || continue
        local_name="$(basename "$pidfile" .pid)"
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            pids+=("$pid")
            echo "stopped $local_name (pid $pid)"
        fi
        rm -f "$pidfile"
    done
    # Wait for the processes to actually exit. A sidecar that is still
    # shutting down answers /ready for a moment, so an immediate `up`
    # (make restart) would skip relaunching it and FlowCat would then fail
    # its startup readiness check against a dead port.
    local i=0
    while [ "${#pids[@]}" -gt 0 ] && [ "$i" -lt 100 ]; do
        local alive=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        [ "${#alive[@]}" -gt 0 ] || break
        pids=("${alive[@]}")
        i=$((i + 1))
        sleep 0.1
    done
    if [ "${#pids[@]}" -gt 0 ] && [ "$i" -ge 100 ]; then
        echo "WARNING: still running after 10s: ${pids[*]} (SIGKILL)" >&2
        kill -9 "${pids[@]}" 2>/dev/null || true
    fi
}

run_tests() {
    (cd "$POC_DIR" && "$VENV/bin/pytest" harness -m smoke "$@")
}

case "${1:-}" in
up) up ;;
down) down ;;
test) shift; run_tests "$@" ;;
*) echo "usage: $0 up|down|test [pytest args]" >&2; exit 2 ;;
esac
