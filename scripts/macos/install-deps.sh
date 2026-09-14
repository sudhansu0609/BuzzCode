#!/usr/bin/env bash
# scripts/macos/install-deps.sh
# Verifies and installs dependencies needed to build and run BuzzCode on macOS with Apple Silicon / Metal.

set -euo pipefail

echo "== buzzcode: macOS dependency check =="

# 1. Check for Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    echo "[!] Xcode Command Line Tools not detected."
    echo "    Installing via xcode-select --install..."
    xcode-select --install || true
    echo "    Please complete the Xcode prompt and re-run this script."
    exit 1
else
    echo "  [ok]   xcode-select installed"
fi

# 2. Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "[!] Homebrew is not installed."
    echo "    Visit https://brew.sh or run:"
    echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
else
    echo "  [ok]   brew $(brew --version | head -n 1)"
fi

# 3. Install build tools: cmake, ninja, ripgrep
NEEDED=()
for tool in cmake ninja rg; do
    if ! command -v "$tool" &>/dev/null; then
        NEEDED+=("$tool")
    fi
done

if [ ${#NEEDED[@]} -gt 0 ]; then
    echo "Installing missing tools: ${NEEDED[*]}..."
    brew install cmake ninja ripgrep
fi

for tool in cmake ninja rg git; do
    if command -v "$tool" &>/dev/null; then
        echo "  [ok]   $tool ($(command -v "$tool"))"
    else
        echo "  [MISSING] $tool"
    fi
done

# 4. llama.cpp / llama-server
# Homebrew has prebuilt, Metal-accelerated llama.cpp:
if ! command -v llama-server &>/dev/null && [ ! -f "/opt/homebrew/bin/llama-server" ] && [ ! -f "/usr/local/bin/llama-server" ]; then
    echo ""
    echo "Tip: You can install prebuilt Metal-accelerated llama.cpp directly via Homebrew:"
    echo "     brew install llama.cpp"
    echo "Or run scripts/macos/build-llama.sh to compile the latest version from source."
else
    SERVER_BIN="$(command -v llama-server 2>/dev/null || echo "/opt/homebrew/bin/llama-server")"
    echo "  [ok]   llama-server ($SERVER_BIN)"
fi

echo ""
echo "All required dependencies present for macOS."
