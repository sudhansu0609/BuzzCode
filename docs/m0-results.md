# M0 / M1 results — engine bring-up and first agent runs (2026-08-23)

Machine: RTX 5060 Ti 16 GB (sm_120, ~448 GB/s), i7-14700K, 31.8 GB RAM, Windows 11.
Desktop apps hold ~2.3 GB VRAM at idle → ~13.4 GB usable.

## Engine

| Item | Result |
|---|---|
| llama.cpp | **b10590** (commit 6657ded4f) |
| Source build with local **CUDA 13.2** | builds fine, **K-quants OK, IQ-quants produce garbage on GPU** (CPU path coherent) → CUDA 13.2 miscompiles IQ kernels, as Unsloth warned |
| **Prebuilt `llama-b10590-bin-win-cuda-13.3-x64`** | all quants coherent; **this is the engine in use** (`engine.provider = "prebuilt"` in `~/.buzzcode/config.toml`) |
| Native sm_120 source build | re-enable after installing CUDA 13.1 or 13.3 toolkit side-by-side (`scripts\build-llama.ps1`) |

## Model: Qwen3.8-27B (arch `qwen35`)

64 layers = 16 full attention (every 4th) + 48 Gated DeltaNet; 4 KV heads × 256 head_dim → **36,992 B/token KV (q8_0) → 1.15 GiB @ 32K**; recurrent state ≈ 148 MiB/seq; MTP head ≈ 251 MiB (`nextn_predict_layers = 1`, the GGUF counts it as block 64).

| Quant | File | Plan | Load | VRAM used | Decode | MTP accept | Prompt cache |
|---|---|---|---|---|---|---|---|
| lmstudio-community **Q4_K_M** | 15.65 GiB | ngl 46/64, ctx 16K (19 layers on CPU) | 11.2 s | 12.1 GiB | 16.4 tok/s | 79 % | 99 % |
| unsloth **UD-IQ3_XXS** (default `qwen38-27b-fast`) | 10.18 GiB | **all 64 layers + output on GPU, ctx 32K** | **4.2 s** | **12.08 GiB** | **44.5 tok/s** (30.7 without MTP) | **83 %** | **99 %** |

Planner after calibration: estimate == measured (+0 %) for IQ3_XXS.

## Chat-template facts (verified with `/apply-template`)

* `reasoning_effort` accepts only **`xhigh` (default), `medium`, `low`** — `high` raises a Jinja exception.
* `xhigh` and `low` **inject a sentence at the start of the system message**; `medium` injects nothing → the harness pins **medium** for the whole session (prefix-stable).
* `enable_thinking:false` only changes the prompt tail (`<think>\n\n</think>`) → used for summaries / commit messages without breaking the cache.
* Tools are rendered into the system message *before* the user's system content; the native call format is `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`; llama-server parses it natively and the harness has a fallback parser for the same syntax.
* `--cache-reuse` is not used (hybrid/recurrent model); LCP prefix reuse + `--ctx-checkpoints` work: turn-2 prompt processing ≈ 300–500 ms.

## First agent runs (headless, `--mode yolo`)

1. "What does this repository do?" — 2 batched read-only calls, correct 3-sentence answer.
2. `eval/tasks/rust-add-fn` — read → edit_file → auto-verify (`cargo check`) **failed** (fixture was nested in the parent cargo workspace) → model diagnosed it, added `[workspace]`, added test → verify ok → `cargo test` 2/2 → check.ps1 exit 0.
   Per-turn: 44–51 tok/s, MTP 80–95 %, cache 77–97 %, prompt 350–1050 ms.

## Benchmarks (`buzzcode bench`, IQ3_XXS, all-GPU, 32K ctx)

| Prompt tokens | prompt tok/s | decode tok/s | TTFT cold | TTFT warm (cached) | MTP accept |
|---|---|---|---|---|---|
| 2,362 | 727 | **46.7** | 3.3 s | **0.43 s** | 81 % |
| 9,082 | 786 | 39.9 | 11.6 s | 0.47 s | 67 % |
| 18,225 | 758 | 39.5 | 24.1 s | 0.52 s | 74 % |

Server VRAM 12.4 GiB · harness RSS **18 MiB** · `bench cache` (6 turns): 96–97 % cached per turn, ~390 ms prompt/turn, prefix stable.

`engine tune` (code-rewrite prompt, 400 tokens): **draft-mtp 44.0 tok/s** (75 % accept) · ngram-simple 33.4 · none 28.9.

`bench eval` (yolo, headless, wall time includes engine start + index + warm-up):

| task | result | turns | wall |
|---|---|---|---|
| rust-add-fn (add fn + test, cargo test) | PASS | 3–4 | 23–25 s |
| rust-fix-compile (fix borrow error, keep API) | PASS | 3 | 26 s |
| py-fix-test (find bug, don't touch tests, pytest) | PASS | 3 | 20 s |
| explain-readonly (no edits, cite test) | PASS | 1 | 14 s |

Subagents: `spawn_agent explore` ran 4 turns in its own context; the parent's next turn still hit 79 % prompt cache (`--cache-ram` restore across the slot switch works).

## Open items

* `/apply-template` requires valid kwargs (500 on `high`) — doctor now tests `xhigh`/`low`/`medium`.
* Planner: checkpoint memory term halved after measurement; still re-validate on other models.
* Prompt-processing throughput reads low on tiny prompts (overhead-dominated); measure with 4K/16K prompts in `bench engine` (M6).
