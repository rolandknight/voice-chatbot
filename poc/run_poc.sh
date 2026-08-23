#!/usr/bin/env bash
# PoC orchestration (poc/CONTRACT.md "Run orchestration"):
#   run_poc.sh up    - start stubs + kokoro shim (+ flowcat-poc if built), wait on healths
#   run_poc.sh test  - run the smoke suite (pytest -m smoke)
#   run_poc.sh down  - stop everything started by `up`
# Logs: poc/logs/<name>.log   PIDs: poc/logs/<name>.pid
set -euo pipefail

POC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$POC_DIR/.venv"
LOGS="$POC_DIR/logs"
FLOWCAT_BIN="$POC_DIR/flowcat/target/debug/flowcat-poc"

# .env provides defaults; already-set caller env wins (same semantics as
# flowcat's dotenvy load, so e.g. OPENROUTER_BASE_URL=... overrides work).
if [ -f "$POC_DIR/.env" ]; then
    while IFS= read -r line; do
        case "$line" in '' | \#*) continue ;; esac
        key="${line%%=*}"
        [ -z "${!key+x}" ] && export "${line?}"
    done <"$POC_DIR/.env"
fi

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

up() {
    mkdir -p "$LOGS"
    start_proc stubs "$POC_DIR/stubs" "$VENV/bin/uvicorn" stub_server:app --host 127.0.0.1 --port 8790
    start_proc kokoro "$POC_DIR/stubs" "$VENV/bin/uvicorn" kokoro_shim:app --host 127.0.0.1 --port 8880
    if [ -x "$FLOWCAT_BIN" ]; then
        start_proc flowcat "$POC_DIR/flowcat" "$FLOWCAT_BIN"
    else
        echo "NOTE: $FLOWCAT_BIN not built yet — start the FlowCat server separately (Rust side)."
    fi
    wait_health stubs http://127.0.0.1:8790/health 30
    wait_health kokoro http://127.0.0.1:8880/health 600 # first start downloads ~350 MB of models
    if [ -x "$FLOWCAT_BIN" ]; then
        wait_health flowcat http://127.0.0.1:6210/healthz 60 ||
            { echo "flowcat /healthz absent; checking TCP..."; timeout 5 bash -c 'until echo >/dev/tcp/127.0.0.1/6210; do sleep 0.5; done' 2>/dev/null && echo "flowcat TCP up"; }
    fi
}

down() {
    for pidfile in "$LOGS"/*.pid; do
        [ -f "$pidfile" ] || continue
        local_name="$(basename "$pidfile" .pid)"
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "stopped $local_name (pid $pid)"
        fi
        rm -f "$pidfile"
    done
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
