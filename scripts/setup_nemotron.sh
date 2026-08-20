#!/usr/bin/env bash
# Install the pinned NeMo-Speech.cpp runtime and English Nemotron streaming
# model into PoC-local directories. Nothing is added to the user's PATH.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POC_DIR="$PROJECT_DIR/poc"

# Read the PoC profile for standalone use while preserving explicit shell or
# Make overrides. run_poc.sh already exports this file before launching us.
DEVICE_OVERRIDE_SET="${POC_NEMOTRON_DEVICE+set}"
DEVICE_OVERRIDE="${POC_NEMOTRON_DEVICE-}"
MODEL_ROOT_OVERRIDE_SET="${NEMO_SPEECH_MODEL_DIR+set}"
MODEL_ROOT_OVERRIDE="${NEMO_SPEECH_MODEL_DIR-}"
if [ -f "$POC_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$POC_DIR/.env"
    set +a
fi
[ -n "$DEVICE_OVERRIDE_SET" ] && POC_NEMOTRON_DEVICE="$DEVICE_OVERRIDE"
[ -n "$MODEL_ROOT_OVERRIDE_SET" ] && NEMO_SPEECH_MODEL_DIR="$MODEL_ROOT_OVERRIDE"

NEMO_SPEECH_VERSION="0.1.0"
NEMO_SPEECH_TAG="v$NEMO_SPEECH_VERSION"
RELEASE_BASE="https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/$NEMO_SPEECH_TAG"
NATIVE_PARENT="$POC_DIR/.deps/nemo-speech"
NATIVE_ROOT="$NATIVE_PARENT/$NEMO_SPEECH_TAG"
MODEL_ROOT="${NEMO_SPEECH_MODEL_DIR:-$POC_DIR/models/nemotron}"
REQUESTED_DEVICE="${POC_NEMOTRON_DEVICE:-auto}"

CLEANUP_DIRS=()

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

safe_remove_dir() {
    local path="$1"
    case "$path" in
    "$NATIVE_PARENT"/.setup-* | "$NATIVE_ROOT".previous-*) rm -rf -- "$path" ;;
    *) fail "refusing to remove unexpected path: $path" ;;
    esac
}

cleanup() {
    local path
    for path in "${CLEANUP_DIRS[@]}"; do
        if [ -e "$path" ]; then
            safe_remove_dir "$path"
        fi
    done
}
trap cleanup EXIT

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        fail "sha256sum or shasum is required to verify NeMo-Speech.cpp"
    fi
}

verify_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256_file "$path")"
    [ "$actual" = "$expected" ] || \
        fail "SHA-256 mismatch for $(basename "$path"): expected $expected, got $actual"
}

select_release() {
    local kernel machine
    kernel="${POC_PLATFORM_OVERRIDE:-$(uname -s)}"
    machine="${POC_ARCH_OVERRIDE:-$(uname -m)}"

    case "$REQUESTED_DEVICE" in
    auto | cpu | cuda:0 | metal) ;;
    *) fail "invalid POC_NEMOTRON_DEVICE '$REQUESTED_DEVICE' (expected auto, cuda:0, cpu, or metal)" ;;
    esac

    case "$kernel/$machine" in
    Linux/x86_64 | Linux/amd64)
        case "$REQUESTED_DEVICE" in
        metal) fail "POC_NEMOTRON_DEVICE=metal is only supported on Apple Silicon macOS" ;;
        cuda:0)
            command -v nvidia-smi >/dev/null 2>&1 || \
                fail "POC_NEMOTRON_DEVICE=cuda:0 requires a working NVIDIA driver (nvidia-smi)"
            nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || \
                fail "the NVIDIA driver is installed but unavailable"
            RELEASE_BACKEND="cuda"
            ;;
        cpu) RELEASE_BACKEND="cpu" ;;
        auto)
            if command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
                RELEASE_BACKEND="cuda"
            else
                RELEASE_BACKEND="cpu"
            fi
            ;;
        esac
        PLATFORM="linux-x86_64"
        case "$RELEASE_BACKEND" in
        cuda) ARCHIVE_SHA256="e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147" ;;
        cpu) ARCHIVE_SHA256="0f74131d631ad2c694cf0ec53490866bb6461147959589a69fb6fc231944065b" ;;
        esac
        ;;
    Darwin/arm64 | Darwin/aarch64)
        case "$REQUESTED_DEVICE" in
        cuda:0) fail "POC_NEMOTRON_DEVICE=cuda:0 is only supported on Linux" ;;
        metal | auto) RELEASE_BACKEND="metal" ;;
        cpu) RELEASE_BACKEND="cpu" ;;
        esac
        PLATFORM="macos-aarch64"
        case "$RELEASE_BACKEND" in
        metal) ARCHIVE_SHA256="f1dff4f9dd9c96214f8cb78b982812459132df8a4ad1a42409fd94de4a366244" ;;
        cpu) ARCHIVE_SHA256="971661d38d4bf97a63c528d13041a964316d25068d8df045e5b4839848092f25" ;;
        esac
        ;;
    Darwin/x86_64 | Darwin/amd64)
        case "$REQUESTED_DEVICE" in
        auto | cpu) RELEASE_BACKEND="cpu" ;;
        metal) fail "POC_NEMOTRON_DEVICE=metal requires Apple Silicon" ;;
        cuda:0) fail "POC_NEMOTRON_DEVICE=cuda:0 is only supported on Linux" ;;
        esac
        PLATFORM="macos-x86_64"
        ARCHIVE_SHA256="042a4612e07460fab6a39b5d862aa1e39d0ac3eaedfdb979f3f5fc12de510c20"
        ;;
    *) fail "unsupported platform '$kernel/$machine' (expected Linux x86_64 or macOS)" ;;
    esac

    ARCHIVE_ASSET="nemo-speech-$NEMO_SPEECH_VERSION-$PLATFORM-$RELEASE_BACKEND.tar.gz"
}

native_install_valid() {
    local root="$1"
    local library

    [ -f "$root/.nemo-speech-release" ] || return 1
    grep -Fxq "version=$NEMO_SPEECH_VERSION" "$root/.nemo-speech-release" || return 1
    grep -Fxq "asset=$ARCHIVE_ASSET" "$root/.nemo-speech-release" || return 1
    grep -Fxq "archive_sha256=$ARCHIVE_SHA256" "$root/.nemo-speech-release" || return 1
    [ -x "$root/bin/nemo-speech" ] || return 1
    [ -s "$root/include/nemo_speech/asr.h" ] || return 1

    case "$PLATFORM" in
    linux-*) library="$root/lib/libnemo_speech_asr_c.so.1" ;;
    macos-*) library="$root/lib/libnemo_speech_asr_c.1.dylib" ;;
    *) return 1 ;;
    esac
    [ -s "$library" ] || return 1
}

atomic_replace_dir() {
    local staged="$1"
    local final="$2"
    local backup=""

    [ -d "$staged" ] || fail "staged install directory is missing: $staged"
    if [ -e "$final" ]; then
        backup="${final}.previous-$$"
        [ ! -e "$backup" ] || fail "backup path already exists: $backup"
        mv "$final" "$backup"
    fi

    if ! mv "$staged" "$final"; then
        if [ -n "$backup" ] && [ -e "$backup" ]; then
            mv "$backup" "$final"
        fi
        fail "could not install staged directory at $final"
    fi

    if [ -n "$backup" ] && [ -e "$backup" ]; then
        safe_remove_dir "$backup"
    fi
}

select_release

for command_name in curl tar awk grep mv mkdir mktemp; do
    command -v "$command_name" >/dev/null 2>&1 || fail "$command_name is required"
done

mkdir -p "$NATIVE_PARENT" "$MODEL_ROOT"

if native_install_valid "$NATIVE_ROOT"; then
    echo "NeMo-Speech.cpp already present ($NEMO_SPEECH_TAG, $PLATFORM/$RELEASE_BACKEND)."
else
    native_work="$(mktemp -d "$NATIVE_PARENT/.setup-${NEMO_SPEECH_TAG}.XXXXXX")"
    CLEANUP_DIRS+=("$native_work")
    archive_part="$native_work/$ARCHIVE_ASSET.part"
    payload="$native_work/payload"

    echo "Downloading NeMo-Speech.cpp $NEMO_SPEECH_TAG for $PLATFORM/$RELEASE_BACKEND ..."
    curl --fail --location --retry 3 --connect-timeout 20 \
        --output "$archive_part" "$RELEASE_BASE/$ARCHIVE_ASSET"
    verify_sha256 "$archive_part" "$ARCHIVE_SHA256"

    mkdir -p "$payload"
    tar -xzf "$archive_part" -C "$payload" --strip-components=1
    printf 'version=%s\nasset=%s\narchive_sha256=%s\n' \
        "$NEMO_SPEECH_VERSION" "$ARCHIVE_ASSET" "$ARCHIVE_SHA256" \
        >"$payload/.nemo-speech-release"

    native_install_valid "$payload" || fail "downloaded NeMo-Speech.cpp runtime is incomplete"
    "$payload/bin/nemo-speech" --version >/dev/null
    atomic_replace_dir "$payload" "$NATIVE_ROOT"
    echo "NeMo-Speech.cpp runtime installed."
fi

export NEMO_SPEECH_MODEL_DIR="$MODEL_ROOT"
echo "Downloading or verifying the Nemotron Speech Streaming English model ..."
"$NATIVE_ROOT/bin/nemo-speech" pull nemotron-en

echo "NeMo-Speech.cpp root: $NATIVE_ROOT"
echo "Nemotron model cache: $MODEL_ROOT"
echo "Device profile:       $REQUESTED_DEVICE ($RELEASE_BACKEND release)"
