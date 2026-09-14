#!/usr/bin/env bash
# setup-mac.sh
# ==============================================================================
# BuzzCode - One-Click Setup for macOS (Apple Silicon / Intel)
#
# Run this script right after cloning the repository:
#   chmod +x setup-mac.sh && ./setup-mac.sh
# ==============================================================================

set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "==================================================="
echo "   BuzzCode - Automated Setup for macOS"
echo "==================================================="
echo -e "${NC}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------------------
# 1. Xcode Command Line Tools
# ------------------------------------------------------------------------------
echo -e "${BOLD}[1/5] Checking Xcode Command Line Tools...${NC}"
if ! xcode-select -p &>/dev/null; then
    echo -e "${YELLOW}  --> Xcode Command Line Tools not detected. Launching installer...${NC}"
    xcode-select --install || true
    echo -e "${YELLOW}  --> Please complete the Apple dialog prompt, then re-run ./setup-mac.sh${NC}"
    exit 1
else
    echo -e "${GREEN}  [ok] Xcode Command Line Tools installed.${NC}"
fi

# ------------------------------------------------------------------------------
# 2. Homebrew Package Manager
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[2/5] Checking Homebrew...${NC}"
if ! command -v brew &>/dev/null; then
    if [ -f "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f "/usr/local/bin/brew" ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    else
        echo -e "${YELLOW}  --> Homebrew not found. Installing Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f "/usr/local/bin/brew" ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
fi

if command -v brew &>/dev/null; then
    echo -e "${GREEN}  [ok] Homebrew $(brew --version | head -n 1)${NC}"
else
    echo -e "${RED}  [!] Homebrew installation could not be completed.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Rust Toolchain
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/5] Checking Rust Toolchain (cargo & rustc)...${NC}"
if ! command -v cargo &>/dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo -e "${YELLOW}  --> Rust not found. Installing via rustup...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
fi

if command -v cargo &>/dev/null; then
    echo -e "${GREEN}  [ok] $(cargo --version)${NC}"
else
    echo -e "${RED}  [!] Rust installation failed. Please install from https://rustup.rs${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. macOS Dependencies & Metal-accelerated llama.cpp
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/5] Installing dependencies via Homebrew...${NC}"

PACKAGES_TO_INSTALL=()
for pkg in cmake ninja ripgrep llama.cpp; do
    if ! brew list "$pkg" &>/dev/null; then
        PACKAGES_TO_INSTALL+=("$pkg")
    fi
done

if [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
    echo -e "${CYAN}  --> Installing: ${PACKAGES_TO_INSTALL[*]}...${NC}"
    brew install "${PACKAGES_TO_INSTALL[@]}"
else
    echo -e "${GREEN}  [ok] cmake, ninja, ripgrep, and llama.cpp already installed.${NC}"
fi

# Ensure all scripts have execute permissions
chmod +x "$ROOT_DIR/build.sh" "$ROOT_DIR"/scripts/macos/*.sh 2>/dev/null || true

# ------------------------------------------------------------------------------
# 5. Build and Install BuzzCode
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[5/5] Building and installing BuzzCode binary...${NC}"
cargo install --path "$ROOT_DIR/crates/buzzcode"

# Ensure ~/.cargo/bin is in ~/.zshrc if not already
if ! echo "$PATH" | grep -q "$HOME/.cargo/bin"; then
    export PATH="$HOME/.cargo/bin:$PATH"
    if [ -f "$HOME/.zshrc" ] && ! grep -q '\.cargo/bin' "$HOME/.zshrc"; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.zshrc"
        echo -e "${CYAN}  --> Added ~/.cargo/bin to your ~/.zshrc${NC}"
    fi
fi

echo -e "\n${BOLD}Validating installation...${NC}"
"$HOME/.cargo/bin/buzzcode" engine doctor --static || true

echo -e "${GREEN}${BOLD}"
echo "==================================================="
echo "   BuzzCode successfully setup on macOS!"
echo "==================================================="
echo -e "${NC}"
echo "To start BuzzCode right now, run:"
echo -e "  ${CYAN}${BOLD}buzzcode${NC}"
echo ""
echo "Or in a specific project directory:"
echo -e "  ${CYAN}${BOLD}buzzcode ~/path/to/my-project${NC}"
echo ""
echo "To download a local coding model (Metal-accelerated):"
echo -e "  ${CYAN}buzzcode models pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF${NC}"
echo ""
