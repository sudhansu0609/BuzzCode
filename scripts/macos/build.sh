#!/usr/bin/env bash
# scripts/macos/build.sh
# Build and install BuzzCode on macOS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==================================================="
echo "  BuzzCode - Rebuild and Install (macOS)"
echo "==================================================="
echo ""

cd "$ROOT_DIR"

echo "[1/2] Building and installing buzzcode binary to ~/.cargo/bin..."
cargo install --path crates/buzzcode

echo ""
echo "[2/2] Validating installation..."
"$HOME/.cargo/bin/buzzcode" engine doctor --static || true

echo ""
echo "==================================================="
echo "  BuzzCode successfully installed!"
echo "  Run 'buzzcode' in your terminal to launch."
echo "==================================================="
