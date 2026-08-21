#!/usr/bin/env bash
# Cross-platform build/doctor entry point for the FlowCat PoC.
#
# The default is deliberately conservative:
#   - Apple Silicon/macOS: Metal when the Metal compiler is available.
#   - Linux/NVIDIA: CUDA only when both the driver and nvcc are available.
#   - Everything else: CPU Whisper.
#
# Select the recognizer with POC_STT_BACKEND=whisper|moonshine|nemotron.
# Whisper can be accelerated with POC_STT_ACCELERATOR=cpu|metal|cuda;
# Nemotron runs in the separately installed NeMo-Speech.cpp sidecar selected by
# POC_NEMOTRON_DEVICE=auto|cuda:0|metal|cpu. POC_PLATFORM_OVERRIDE is intended
# for tests of detection logic; normal callers should leave it unset.
set -euo pipefail

POC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$POC_DIR/.." && pwd)"
FLOWCAT_MANIFEST="$POC_DIR/flowcat/Cargo.toml"
# Read the project profile, but let explicit shell/Make overrides win. Keep this
# Bash-3-compatible for the stock shell on older macOS installations.
STT_BACKEND_OVERRIDE_SET="${POC_STT_BACKEND+set}"
STT_BACKEND_OVERRIDE="${POC_STT_BACKEND-}"
STT_ACCELERATOR_OVERRIDE_SET="${POC_STT_ACCELERATOR+set}"
STT_ACCELERATOR_OVERRIDE="${POC_STT_ACCELERATOR-}"
MOONSHINE_HOME_OVERRIDE_SET="${POC_MOONSHINE_HOME+set}"
MOONSHINE_HOME_OVERRIDE="${POC_MOONSHINE_HOME-}"
MOONSHINE_MODEL_OVERRIDE_SET="${POC_MOONSHINE_MODEL+set}"
MOONSHINE_MODEL_OVERRIDE="${POC_MOONSHINE_MODEL-}"
NEMOTRON_DEVICE_OVERRIDE_SET="${POC_NEMOTRON_DEVICE+set}"
NEMOTRON_DEVICE_OVERRIDE="${POC_NEMOTRON_DEVICE-}"
if [ -f "$POC_DIR/.env" ]; then
    set -a
    . "$POC_DIR/.env"
    set +a
fi
[ -n "$STT_BACKEND_OVERRIDE_SET" ] && POC_STT_BACKEND="$STT_BACKEND_OVERRIDE"
[ -n "$STT_ACCELERATOR_OVERRIDE_SET" ] && POC_STT_ACCELERATOR="$STT_ACCELERATOR_OVERRIDE"
[ -n "$MOONSHINE_HOME_OVERRIDE_SET" ] && POC_MOONSHINE_HOME="$MOONSHINE_HOME_OVERRIDE"
[ -n "$MOONSHINE_MODEL_OVERRIDE_SET" ] && POC_MOONSHINE_MODEL="$MOONSHINE_MODEL_OVERRIDE"
[ -n "$NEMOTRON_DEVICE_OVERRIDE_SET" ] && POC_NEMOTRON_DEVICE="$NEMOTRON_DEVICE_OVERRIDE"
PLATFORM="${POC_PLATFORM_OVERRIDE:-$(uname -s)}"
ARCH="$(uname -m)"
STT_BACKEND="${POC_STT_BACKEND:-whisper}"
REQUESTED_ACCELERATOR="${POC_STT_ACCELERATOR:-auto}"
ACCELERATOR=""
ACCELERATOR_REASON=""
OPUS_SOURCE=""
MOONSHINE_HOME="${POC_MOONSHINE_HOME:-$POC_DIR/.deps/moonshine/v0.1.3}"
MOONSHINE_MODEL="${POC_MOONSHINE_MODEL:-$POC_DIR/models/moonshine/download.moonshine.ai/model/medium-streaming-en/quantized_26_07_30}"
NEMOTRON_HOME="${POC_NEMOTRON_HOME:-$POC_DIR/.deps/nemo-speech/v0.1.0}"
NEMOTRON_MODEL_ROOT="${NEMO_SPEECH_MODEL_DIR:-$POC_DIR/models/nemotron}"
NEMOTRON_MODEL="$NEMOTRON_MODEL_ROOT/nvidia/nemotron-speech-streaming-en-0.6b/ebe59e5a817142986528bbbee5dba8db7b38ed50/nemotron-speech-streaming-en-0.6b.q8_0.gguf"
NEMOTRON_DEVICE="${POC_NEMOTRON_DEVICE:-auto}"
WITH_MOONSHINE="${POC_WITH_MOONSHINE:-auto}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

detect_accelerator() {
    case "$STT_BACKEND" in
    whisper) ;;
    moonshine)
        case "$REQUESTED_ACCELERATOR" in
        auto | cpu) ;;
        *) fail "POC_STT_ACCELERATOR=$REQUESTED_ACCELERATOR applies only to Whisper; Moonshine uses its native CPU runtime" ;;
        esac
        ACCELERATOR="cpu"
        ACCELERATOR_REASON="Moonshine native CPU runtime"
        return
        ;;
    nemotron | nvidia)
        case "$NEMOTRON_DEVICE" in
        auto)
            case "$PLATFORM" in
            Darwin)
                if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
                    ACCELERATOR="metal"
                    ACCELERATOR_REASON="NeMo-Speech.cpp auto-selects Apple Metal"
                else
                    ACCELERATOR="cpu"
                    ACCELERATOR_REASON="NeMo-Speech.cpp uses CPU on Intel macOS"
                fi
                ;;
            Linux)
                if command -v nvidia-smi >/dev/null 2>&1 \
                    && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
                    ACCELERATOR="cuda"
                    ACCELERATOR_REASON="NeMo-Speech.cpp auto-selects the NVIDIA GPU"
                else
                    ACCELERATOR="cpu"
                    ACCELERATOR_REASON="NeMo-Speech.cpp auto-selects portable CPU"
                fi
                ;;
            *) fail "unsupported platform '$PLATFORM' (supported: Darwin, Linux)" ;;
            esac
            ;;
        cuda:0)
            [ "$PLATFORM" = "Linux" ] || fail "POC_NEMOTRON_DEVICE=cuda:0 is only supported on Linux"
            command -v nvidia-smi >/dev/null 2>&1 || fail "POC_NEMOTRON_DEVICE=cuda:0 requires nvidia-smi"
            ACCELERATOR="cuda"; ACCELERATOR_REASON="explicit POC_NEMOTRON_DEVICE=cuda:0"
            ;;
        metal)
            [ "$PLATFORM" = "Darwin" ] || fail "POC_NEMOTRON_DEVICE=metal is only supported on macOS"
            { [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; } || fail "POC_NEMOTRON_DEVICE=metal requires Apple Silicon"
            ACCELERATOR="metal"; ACCELERATOR_REASON="explicit POC_NEMOTRON_DEVICE=metal"
            ;;
        cpu) ACCELERATOR="cpu"; ACCELERATOR_REASON="explicit POC_NEMOTRON_DEVICE=cpu" ;;
        *) fail "invalid POC_NEMOTRON_DEVICE '$NEMOTRON_DEVICE' (expected auto, cuda:0, metal, or cpu)" ;;
        esac
        return
        ;;
    *) fail "invalid POC_STT_BACKEND '$STT_BACKEND' (expected whisper, moonshine, or nemotron)" ;;
    esac

    case "$REQUESTED_ACCELERATOR" in
    auto)
        case "$PLATFORM" in
        Darwin)
            if command -v xcrun >/dev/null 2>&1 && xcrun --find metal >/dev/null 2>&1; then
                ACCELERATOR="metal"
                ACCELERATOR_REASON="macOS Metal toolchain detected"
            else
                ACCELERATOR="cpu"
                ACCELERATOR_REASON="Metal compiler unavailable; using portable CPU Whisper"
            fi
            ;;
        Linux)
            if command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 \
                && command -v nvcc >/dev/null 2>&1; then
                ACCELERATOR="cuda"
                ACCELERATOR_REASON="NVIDIA driver and CUDA toolkit detected"
            elif command -v nvidia-smi >/dev/null 2>&1; then
                ACCELERATOR="cpu"
                ACCELERATOR_REASON="NVIDIA driver detected but nvcc is unavailable; using CPU Whisper"
            else
                ACCELERATOR="cpu"
                ACCELERATOR_REASON="no CUDA toolchain detected; using portable CPU Whisper"
            fi
            ;;
        *)
            fail "unsupported platform '$PLATFORM' (supported: Darwin, Linux)"
            ;;
        esac
        ;;
    cpu)
        ACCELERATOR="cpu"
        ACCELERATOR_REASON="explicit POC_STT_ACCELERATOR=cpu"
        ;;
    metal)
        [ "$PLATFORM" = "Darwin" ] || fail "Metal Whisper is only supported on macOS"
        command -v xcrun >/dev/null 2>&1 || fail "xcrun is required for Metal Whisper"
        xcrun --find metal >/dev/null 2>&1 || fail "Metal compiler not found; install Xcode or use POC_STT_ACCELERATOR=cpu"
        ACCELERATOR="metal"
        ACCELERATOR_REASON="explicit POC_STT_ACCELERATOR=metal"
        ;;
    cuda)
        [ "$PLATFORM" = "Linux" ] || fail "this PoC enables CUDA Whisper only on Linux"
        command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required for CUDA Whisper"
        command -v nvcc >/dev/null 2>&1 || fail "nvcc is required for CUDA Whisper; install a matching CUDA toolkit or use POC_STT_ACCELERATOR=cpu"
        ACCELERATOR="cuda"
        ACCELERATOR_REASON="explicit POC_STT_ACCELERATOR=cuda"
        ;;
    *)
        fail "invalid POC_STT_ACCELERATOR '$REQUESTED_ACCELERATOR' (expected auto, cpu, metal, or cuda)"
        ;;
    esac
}

configure_opus() {
    command -v pkg-config >/dev/null 2>&1 || {
        case "$PLATFORM" in
        Darwin) fail "pkg-config is required; install it with: brew install pkg-config opus" ;;
        Linux) fail "pkg-config is required; install it with your package manager (for example: sudo apt install pkg-config libopus-dev)" ;;
        esac
    }

    if pkg-config --exists opus 2>/dev/null; then
        OPUS_SOURCE="system ($(pkg-config --modversion opus))"
        return
    fi

    local bundled_pc="$POC_DIR/.deps/prefix/lib/pkgconfig/opus.pc"
    if [ -f "$bundled_pc" ]; then
        export PKG_CONFIG_PATH="$(dirname "$bundled_pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
        if pkg-config --exists opus 2>/dev/null; then
            OPUS_SOURCE="PoC-local ($(pkg-config --modversion opus))"
            return
        fi
    fi

    if command -v autoreconf >/dev/null 2>&1 && command -v make >/dev/null 2>&1; then
        OPUS_SOURCE="audiopus-sys bundled source (autotools)"
        return
    fi

    case "$PLATFORM" in
    Darwin)
        fail "Opus development files not found; run: brew install pkg-config opus"
        ;;
    Linux)
        fail "Opus development files not found; on Debian/Ubuntu/Pop!_OS run: sudo apt install libopus-dev"
        ;;
    esac
}

print_plan() {
    echo "PoC platform:      $PLATFORM/$ARCH"
    echo "STT backend:       $STT_BACKEND"
    echo "STT accelerator:   $ACCELERATOR ($ACCELERATOR_REASON)"
    echo "Opus:              $OPUS_SOURCE"
}

moonshine_runtime_ready() {
    [ -s "$MOONSHINE_HOME/include/moonshine-c-api.h" ] \
        && [ -s "$MOONSHINE_MODEL/streaming_config.json" ] \
        && case "$PLATFORM" in
            Darwin) [ -s "$MOONSHINE_HOME/lib/libmoonshine.a" ] ;;
            Linux) [ -s "$MOONSHINE_HOME/lib/libmoonshine.so" ] ;;
            *) false ;;
        esac
}

doctor() {
    local failed=0

    for command_name in cargo python3 curl cmake; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "MISSING: $command_name" >&2
            failed=1
        fi
    done

    if [ ! -f "$POC_DIR/.env" ]; then
        echo "MISSING: poc/.env (copy poc/.env.example and add OPENROUTER_API_KEY)" >&2
        failed=1
    elif ! grep -Eq '^OPENROUTER_API_KEY=.+$' "$POC_DIR/.env"; then
        echo "MISSING: non-empty OPENROUTER_API_KEY in poc/.env" >&2
        failed=1
    fi

    local required_files=("$POC_DIR/models/silero_vad.onnx")
    if [ "$STT_BACKEND" = "whisper" ]; then
        required_files+=("$POC_DIR/models/ggml-base.en.bin")
    elif [ "$STT_BACKEND" = "moonshine" ]; then
        required_files+=(
            "$MOONSHINE_HOME/include/moonshine-c-api.h"
            "$MOONSHINE_MODEL/streaming_config.json"
            "$MOONSHINE_MODEL/tokenizer.bin"
        )
        case "$PLATFORM" in
        Darwin) required_files+=("$MOONSHINE_HOME/lib/libmoonshine.a") ;;
        Linux) required_files+=("$MOONSHINE_HOME/lib/libmoonshine.so") ;;
        esac
    else
        required_files+=(
            "$NEMOTRON_HOME/bin/nemo-speech"
            "$NEMOTRON_MODEL"
        )
    fi

    for required_file in "${required_files[@]}"; do
        if [ ! -s "$required_file" ]; then
            if [ "$STT_BACKEND" = "moonshine" ]; then
                echo "MISSING: ${required_file#"$REPO_DIR/"} (run ./scripts/setup_moonshine.sh)" >&2
            elif [ "$STT_BACKEND" = "nemotron" ] || [ "$STT_BACKEND" = "nvidia" ]; then
                echo "MISSING: ${required_file#"$REPO_DIR/"} (run ./scripts/setup_nemotron.sh)" >&2
            else
                echo "MISSING: ${required_file#"$REPO_DIR/"} (run make poc-setup)" >&2
            fi
            failed=1
        fi
    done

    print_plan
    [ "$failed" -eq 0 ] || exit 1
    echo "PoC prerequisites: ready"
}

build() {
    local cargo_args=(build --manifest-path "$FLOWCAT_MANIFEST")
    local cargo_features=()
    local profile_file="$POC_DIR/logs/build-profile.env"
    local profile_tmp="$profile_file.tmp"
    local include_moonshine=0
    case "$WITH_MOONSHINE" in
    1 | true | yes) include_moonshine=1 ;;
    0 | false | no) ;;
    auto)
        if [ "$STT_BACKEND" = "moonshine" ] || moonshine_runtime_ready; then
            include_moonshine=1
        fi
        ;;
    *) fail "invalid POC_WITH_MOONSHINE '$WITH_MOONSHINE' (expected auto, 1, or 0)" ;;
    esac
    if [ "$STT_BACKEND" = "moonshine" ]; then
        include_moonshine=1
    fi
    if [ "$include_moonshine" -eq 1 ]; then
        moonshine_runtime_ready || fail "Moonshine support was requested but its runtime/model is incomplete; run ./scripts/setup_moonshine.sh"
        export MOONSHINE_LIB_DIR="${MOONSHINE_LIB_DIR:-$MOONSHINE_HOME/lib}"
        cargo_features+=(moonshine)
    fi
    if [ "$STT_BACKEND" = "whisper" ] && [ "$ACCELERATOR" != "cpu" ]; then
        cargo_features+=("$ACCELERATOR")
    fi
    if [ "${#cargo_features[@]}" -gt 0 ]; then
        local features_csv
        features_csv="$(IFS=,; echo "${cargo_features[*]}")"
        cargo_args+=(--features "$features_csv")
    fi

    print_plan
    cargo "${cargo_args[@]}"
    mkdir -p "$POC_DIR/logs"
    {
        printf 'POC_BUILD_PLATFORM=%s\n' "$PLATFORM/$ARCH"
        printf 'POC_STT_BACKEND=%s\n' "$STT_BACKEND"
        printf 'POC_STT_ACCELERATOR=%s\n' "$ACCELERATOR"
        printf 'POC_OPUS_SOURCE=%s\n' "$OPUS_SOURCE"
        printf 'POC_CARGO_FEATURES=%s\n' "${features_csv:-none}"
    } >"$profile_tmp"
    mv "$profile_tmp" "$profile_file"
    echo "Build profile:     ${profile_file#"$REPO_DIR/"}"
}

main() {
    local command_name="${1:-doctor}"
    detect_accelerator
    configure_opus

    case "$command_name" in
    plan) print_plan ;;
    doctor) doctor ;;
    build) build ;;
    *) fail "usage: $0 [plan|doctor|build]" ;;
    esac
}

main "$@"
