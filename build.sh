#!/usr/bin/env bash
# build.sh
# Top-level build script for macOS / Linux. Delegates to scripts/macos/build.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/macos/build.sh" "$@"
