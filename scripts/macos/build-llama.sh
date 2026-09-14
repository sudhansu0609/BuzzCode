#!/usr/bin/env bash
# scripts/macos/build-llama.sh
# Clone, pin, and build llama.cpp with Apple Metal support for macOS.

set -euo pipefail

DIR="${1:-$HOME/llama/llama.cpp}"
PIN="${2:-}"
BUILD_DIR="$DIR/build"
JOBS="$(sysctl -n hw.ncpu || echo 8)"

echo "== llama.cpp build (macOS Metal) =="
echo "  dir   : $DIR"
echo "  jobs  : $JOBS"

# Clone if not already present
if [ ! -d "$DIR/.git" ]; then
    mkdir -p "$(dirname "$DIR")"
    git clone https://github.com/ggml-org/llama.cpp "$DIR"
fi

git -C "$DIR" fetch --tags origin

if [ -z "$PIN" ]; then
    PIN="$(git -C "$DIR" tag --list "b*" --sort=-v:refname | head -n 1)"
    echo "  pin   : $PIN (latest tag)"
else
    echo "  pin   : $PIN"
fi

git -C "$DIR" checkout --quiet "$PIN"
COMMIT="$(git -C "$DIR" rev-parse --short HEAD)"
echo "  commit: $COMMIT"

# Configure with Metal enabled
mkdir -p "$BUILD_DIR"
cmake -S "$DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF

# Build
cmake --build "$BUILD_DIR" --config Release --target llama-server llama-bench llama-cli -j "$JOBS"

SERVER="$BUILD_DIR/bin/llama-server"
if [ ! -f "$SERVER" ]; then
    SERVER="$(find "$BUILD_DIR" -name "llama-server" -type f -perm +111 | head -n 1)"
fi

if [ -z "$SERVER" ] || [ ! -f "$SERVER" ]; then
    echo "Error: llama-server binary was not produced."
    exit 1
fi

# Record in ~/.buzzcode/engine/build-info.toml
INFO_DIR="$HOME/.buzzcode/engine"
mkdir -p "$INFO_DIR"
cat <<EOF > "$INFO_DIR/build-info.toml"
# written by scripts/macos/build-llama.sh
dir = "$DIR"
pin = "$PIN"
commit = "$COMMIT"
cuda_path = ""
cuda_arch = ""
server = "$SERVER"
built_at = "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
EOF

echo "Built: $SERVER"
echo "Recorded: $INFO_DIR/build-info.toml"
"$SERVER" --version || true
