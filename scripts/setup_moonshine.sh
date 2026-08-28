#!/usr/bin/env bash
# Install the pinned Moonshine native runtime and Medium Streaming English model.
#
# The native release is staged and SHA-256 verified before it becomes visible.
# Moonshine's official downloader stages model files with size and CRC32C checks;
# this script then moves the completed cache into its deterministic PoC path.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POC_DIR="$PROJECT_DIR" # runtime root (models/, .deps/, .env); the PoC trees are archived

MOONSHINE_VERSION="v0.1.3"
MOONSHINE_PACKAGE_VERSION="${MOONSHINE_VERSION#v}"
MOONSHINE_MODEL_ARCH="5" # MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING
MOONSHINE_MODEL_REL="download.moonshine.ai/model/medium-streaming-en/quantized_26_07_30"

NATIVE_PARENT="$POC_DIR/.deps/moonshine"
NATIVE_ROOT="$NATIVE_PARENT/$MOONSHINE_VERSION"
MODEL_PARENT="$POC_DIR/models"
MODEL_ROOT="$MODEL_PARENT/moonshine"
MODEL_DIR="$MODEL_ROOT/$MOONSHINE_MODEL_REL"
RELEASE_BASE="https://github.com/moonshine-ai/moonshine/releases/download/$MOONSHINE_VERSION"

CLEANUP_DIRS=()

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

safe_remove_dir() {
    local path="$1"

    # Only remove staging/backup directories created by this script. Never let
    # an empty or unexpectedly broad path reach rm -rf.
    case "$path" in
    "$NATIVE_PARENT"/.setup-* | "$NATIVE_ROOT".previous-* | \
        "$MODEL_PARENT"/.moonshine-setup-* | "$MODEL_ROOT".previous-*)
        rm -rf -- "$path"
        ;;
    *)
        fail "refusing to remove unexpected path: $path"
        ;;
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
        fail "sha256sum or shasum is required to verify the Moonshine release"
    fi
}

verify_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256_file "$path")"
    if [ "$actual" != "$expected" ]; then
        fail "SHA-256 mismatch for $(basename "$path"): expected $expected, got $actual"
    fi
}

file_has_size() {
    local path="$1"
    local expected="$2"
    local actual
    [ -f "$path" ] || return 1
    actual="$(wc -c <"$path" | tr -d '[:space:]')"
    [ "$actual" = "$expected" ]
}

native_install_valid() {
    local root="$1"
    local library

    [ -f "$root/.moonshine-release" ] || return 1
    grep -Fxq "version=$MOONSHINE_VERSION" "$root/.moonshine-release" || return 1
    grep -Fxq "asset=$ARCHIVE_ASSET" "$root/.moonshine-release" || return 1
    grep -Fxq "archive_sha256=$ARCHIVE_SHA256" "$root/.moonshine-release" || return 1
    [ -s "$root/include/moonshine-c-api.h" ] || return 1

    case "$PLATFORM" in
    linux-*) library="$root/lib/libmoonshine.so" ;;
    macos-universal) library="$root/lib/libmoonshine.a" ;;
    *) return 1 ;;
    esac
    [ -s "$library" ] || return 1

    if [ "$PLATFORM" = "linux-x86_64" ] || [ "$PLATFORM" = "linux-arm64" ]; then
        [ -s "$root/lib/libonnxruntime.so.1" ] || return 1
    fi
}

model_install_valid() {
    local root="$1"
    local dir="$root/$MOONSHINE_MODEL_REL"

    [ -f "$root/.moonshine-release" ] || return 1
    grep -Fxq "version=$MOONSHINE_VERSION" "$root/.moonshine-release" || return 1
    grep -Fxq "model_arch=$MOONSHINE_MODEL_ARCH" "$root/.moonshine-release" || return 1

    # Sizes come from the dependency catalog embedded in Moonshine v0.1.3.
    # The official downloader verifies CRC32C before this marker is written;
    # checking sizes here makes subsequent setup runs cheap but still robust
    # against truncated or accidentally replaced cache files.
    file_has_size "$dir/adapter.ort" 3651296 || return 1
    file_has_size "$dir/cross_kv.ort" 11643776 || return 1
    file_has_size "$dir/decoder_kv.ort" 146972408 || return 1
    file_has_size "$dir/encoder.ort" 94705376 || return 1
    file_has_size "$dir/frontend.ort" 47467576 || return 1
    file_has_size "$dir/streaming_config.json" 513 || return 1
    file_has_size "$dir/tokenizer.bin" 249974 || return 1
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

case "$(uname -s)" in
Linux)
    case "$(uname -m)" in
    x86_64 | amd64)
        PLATFORM="linux-x86_64"
        ARCHIVE_ASSET="moonshine-voice-linux-x86_64.tar.gz"
        ARCHIVE_SHA256="6bbd4b0ecccebbfabd43527ffbd1857bb1d5bfc1bf920706c7e043057e46fc4d"
        ;;
    arm64 | aarch64)
        PLATFORM="linux-arm64"
        ARCHIVE_ASSET="moonshine-voice-linux-arm64.tar.gz"
        ARCHIVE_SHA256="5ac86ccb05385a25c9e636e3b8de43fcccac7ef293ae3a4d04ff6e5cd9d1161b"
        ;;
    *) fail "unsupported Linux architecture: $(uname -m)" ;;
    esac
    ;;
Darwin)
    case "$(uname -m)" in
    x86_64 | arm64 | aarch64)
        # Despite the asset name, v0.1.3's libmoonshine.a is universal and
        # contains both x86_64 and arm64 slices.
        PLATFORM="macos-universal"
        ARCHIVE_ASSET="moonshine-voice-macos-arm64.tar.gz"
        ARCHIVE_SHA256="cc6604e8f0de5800d831b22d337fa419786184648c356551f0fd9dc851fba2af"
        ;;
    *) fail "unsupported macOS architecture: $(uname -m)" ;;
    esac
    ;;
*) fail "unsupported operating system: $(uname -s) (expected Linux or macOS)" ;;
esac

for command_name in curl tar awk tr wc grep mv mkdir mktemp; do
    command -v "$command_name" >/dev/null 2>&1 || fail "$command_name is required"
done
command -v uv >/dev/null 2>&1 || fail "uv is required for Moonshine's official model downloader"

mkdir -p "$NATIVE_PARENT" "$MODEL_PARENT"

if native_install_valid "$NATIVE_ROOT"; then
    echo "Moonshine native runtime already present ($MOONSHINE_VERSION, $PLATFORM)."
else
    native_work="$(mktemp -d "$NATIVE_PARENT/.setup-${MOONSHINE_VERSION}.XXXXXX")"
    CLEANUP_DIRS+=("$native_work")
    archive_part="$native_work/$ARCHIVE_ASSET.part"
    archive_path="$native_work/$ARCHIVE_ASSET"
    native_payload="$native_work/payload"

    echo "Downloading Moonshine $MOONSHINE_VERSION native runtime for $PLATFORM ..."
    curl --fail --location --retry 3 --connect-timeout 20 \
        --output "$archive_part" "$RELEASE_BASE/$ARCHIVE_ASSET"
    verify_sha256 "$archive_part" "$ARCHIVE_SHA256"
    mv "$archive_part" "$archive_path"

    mkdir -p "$native_payload"
    tar -xzf "$archive_path" -C "$native_payload" --strip-components=1
    printf 'version=%s\nasset=%s\narchive_sha256=%s\n' \
        "$MOONSHINE_VERSION" "$ARCHIVE_ASSET" "$ARCHIVE_SHA256" \
        >"$native_payload/.moonshine-release"

    native_install_valid "$native_payload" || fail "downloaded native runtime is incomplete"
    if [ "$PLATFORM" = "macos-universal" ] && command -v lipo >/dev/null 2>&1; then
        lipo -verify_arch x86_64 arm64 "$native_payload/lib/libmoonshine.a" \
            || fail "Moonshine macOS library is not universal"
    fi
    atomic_replace_dir "$native_payload" "$NATIVE_ROOT"
    echo "Moonshine native runtime installed."
fi

if model_install_valid "$MODEL_ROOT"; then
    echo "Moonshine Medium Streaming English model already present."
else
    model_work="$(mktemp -d "$MODEL_PARENT/.moonshine-setup-${MOONSHINE_VERSION}.XXXXXX")"
    CLEANUP_DIRS+=("$model_work")
    model_payload="$model_work/payload"
    mkdir -p "$model_payload"

    echo "Downloading Moonshine Medium Streaming English model (arch $MOONSHINE_MODEL_ARCH) ..."
    uv run --isolated --no-project \
        --with "moonshine-voice==$MOONSHINE_PACKAGE_VERSION" \
        moonshine-voice download \
        --stt --language en --model-arch "$MOONSHINE_MODEL_ARCH" --root "$model_payload"

    printf 'version=%s\nmodel_arch=%s\n' \
        "$MOONSHINE_VERSION" "$MOONSHINE_MODEL_ARCH" \
        >"$model_payload/.moonshine-release"
    model_install_valid "$model_payload" || fail "downloaded Moonshine model is incomplete"
    atomic_replace_dir "$model_payload" "$MODEL_ROOT"
    echo "Moonshine model installed."
fi

echo "Moonshine native root: $NATIVE_ROOT"
echo "Moonshine model dir:   $MODEL_DIR"
