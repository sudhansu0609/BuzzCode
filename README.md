# buzzcode

A Claude-Code-style terminal coding agent that runs **entirely on a local model** via our own
managed `llama-server`, tuned for a 16 GB GPU + 32 GB RAM (RTX 5060 Ti / i7-14700K).

* Rust workspace, ~18 MiB RSS harness, single static binary.
* Engine manager: GGUF inspection → VRAM planner → spawns/supervises `llama-server`, auto-tunes,
  restarts on crash, warms the KV cache with the session's frozen prefix.
* Prefix-stable context: every request is byte-identical up to the newest message → 96–99 %
  prompt-cache reuse per turn (~0.4 s prompt processing on turn 2+).
* Qwen3.8-27B with MTP speculative decoding: **~44 tok/s** decode, 80–90 % draft acceptance.
* Tools: read/write/edit (exact-string), glob, grep (ripgrep), list_dir, shell, git, tree-sitter
  `outline` / `symbol_search` / PageRank `repo_map`, `spawn_agent` subagents, `write_plan`, MCP.
* Edit-verify loop: after every edit the harness runs your configured checks (`cargo check`,
  `ruff`, `tsc`…) and feeds failures back to the model.
* TUI (ratatui) with permission prompts + diff previews, plan/act modes, slash commands;
  headless `-p` mode with JSON-lines output for scripting and evals.

See `docs/PLAN.md` (architecture) and `docs/m0-results.md` (measurements).

## Quick start (Windows)

```powershell
# 1. toolchain: cmake/ninja/ripgrep (only needed to build llama.cpp from source)
powershell -ExecutionPolicy Bypass -File scripts\install-deps.ps1

# 2. engine: prebuilt CUDA binaries (recommended on this machine — see docs/m0-results.md)
cargo run --release -- engine prebuilt
#    or build from source (needs CUDA 13.1 / 13.3+, NOT 13.2):
#    powershell -ExecutionPolicy Bypass -File scripts\build-llama.ps1

# 3. config (optional): ~/.buzzcode/config.toml; project overrides in .buzzcode/config.toml
cargo run --release -- config --init

# 4. validate everything (downloads the default model on first run)
cargo run --release -- engine doctor

# 5. use it
cargo run --release                                 # TUI in the current project
cargo run --release -- -p "explain src/main.rs"     # headless
cargo run --release -- --mode yolo --json -p "..."  # no prompts, JSON-lines events
```

Install once with `cargo install --path crates/buzzcode` (or run `./build.bat`), then just `buzzcode`.

## Quick start (macOS)

After cloning the repository on your Mac, run the automated setup script:

```bash
chmod +x setup-mac.sh
./setup-mac.sh
```

This single command automatically:
1. Verifies Xcode Command Line Tools and Homebrew.
2. Installs `cmake`, `ninja`, `ripgrep`, and Apple Metal-accelerated `llama.cpp`.
3. Compiles and installs `buzzcode` to `~/.cargo/bin/buzzcode`.
4. Runs `buzzcode engine doctor` to validate your Mac's Apple Silicon GPU / Unified Memory setup.

Once complete, launch BuzzCode anywhere with:
```bash
buzzcode
```

## Commands

| Command | What |
|---|---|
| `buzzcode` | interactive TUI (`/help` for keys) |
| `buzzcode -p "…" [--json] [--mode ask\|edits\|yolo]` | headless task |
| `buzzcode engine doctor [--static] [--keep]` | toolchain/GPU/model/plan/live checks |
| `buzzcode engine plan` | show the VRAM plan + exact `llama-server` command line |
| `buzzcode engine serve` | run the server in the foreground (for other clients) |
| `buzzcode engine build \| prebuilt \| tune \| inspect \| models \| status` | engine management |
| `buzzcode index scan \| map \| search \| outline` | code intelligence without the engine |
| `buzzcode bench engine \| cache \| eval` | throughput, prefix-cache regression, eval tasks |

TUI: `/plan [task]` → investigate read-only and produce a plan → `/act` implements it in a fresh
context. `/mode`, `/effort`, `/compact`, `/cost`, `/engine status|log|restart`, `/new`.

## Layout

```
crates/buzzcode      CLI + TUI wiring, bootstrap, headless, bench commands
crates/cb-config     TOML schema + layering
crates/cb-tool-api   Tool trait / ToolSpec / permissions (dependency-light)
crates/cb-engine     GGUF parser, VRAM planner, llama-server supervisor, SSE client, tuning
crates/cb-core       messages, prefix-stable context, parser, guards, agent loop, subagents, plan
crates/cb-tools      builtin tools
crates/cb-codeintel  tree-sitter index (redb), PageRank repo map, outline/symbol tools
crates/cb-mcp        MCP stdio client → tools
crates/cb-tui        ratatui UI
eval/tasks/*         local eval fixtures (repo/, task.md, check.ps1)
scripts/             install-deps.ps1, build-llama.ps1
```

## Model notes (Qwen3.8-27B)

* Default profile `qwen38-27b-fast`: `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS, 32K ctx, q8_0 KV,
  flash-attn, `--spec-type draft-mtp`, all layers on GPU.
* `reasoning_effort` must be `xhigh|medium|low`; the harness pins **medium** for the session
  because `xhigh`/`low` inject text into the system prompt (prompt-cache break). Summaries and
  commit messages use `enable_thinking:false` (tail-only change, cache-safe).
* `--cache-reuse` is not used (hybrid DeltaNet model); prefix reuse + context checkpoints are.
