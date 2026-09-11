# buzzcode — Local-Model CLI Coding Agent Harness (Rust)

## Context

Goal: a Claude-Code-style terminal coding agent that runs **entirely on local models**, tuned hard for this machine — **RTX 5060 Ti 16GB (Blackwell sm_120, ~448 GB/s)**, **i7-14700K (8P+12E)**, **31.8GB RAM**, Windows 11 — with no quality sacrifice and tight memory management. Project dir `B:\youtubeProjects\Buzzcaf_Media\codeBuzz` is empty (fresh start, no git).

**Decisions already made (user):** Rust · primary model **Qwen3.8-27B dense** · llama.cpp **built from source** · full v1 scope (agent loop + tools, tree-sitter code intel, subagents/plan mode/git, MCP client).

**Answer to "do we need LM Studio or Ollama?"** — **No.** The harness spawns and supervises its own `llama-server.exe` child process (full control over KV cache, MTP speculative decoding, tensor offload, prompt caching). LM Studio and Ollama are just wrappers around the same llama.cpp; they'd cost us the knobs that matter. We *do* reuse your existing LM Studio GGUFs (config points at `C:\Users\singh\.lmstudio\models`) — no re-download needed for Qwen3.8-27B-Q4_K_M, Devstral-Small-2, Gemma-4-12B.

### Verified facts that shape the design
- **Qwen3.8-27B** (released 2026-08-14): 64 layers = 16 full-attention + 48 Gated DeltaNet (linear) → KV cache ~1GB (q8) / ~2GB (f16) at 32K ctx. GGUF ships an **MTP head** → `--spec-type draft-mtp --spec-draft-n-max 2` gives +33–60% decode (4090: 47→76 tok/s). Requires **`--parallel 1`** (single slot). Hybrid/recurrent arch ⇒ **`--cache-reuse` does NOT work**; only longest-common-prefix reuse works ⇒ **prompt-prefix byte-stability is the #1 harness design constraint**. Needs llama.cpp ≥ ~b10450 (DeltaNet CUDA kernels were broken before commit `ece963f41`).
- **Quant sizes**: Q4_K_M = 15.7GB (on disk already — too big for 16GB), UD-Q4_K_XL ≈ 16–19GB, UD-Q3_K_XL ≈ 12–14GB, IQ3_XXS/Q3_K_S ≈ 10.5–11.5GB.
- **Real free VRAM is ~13.7GB** (desktop apps — Chrome, Adobe Animate, buzzanimate.exe — hold ~2.3GB at idle). Plan budgets against *free*, not total.
- **Toolchain gaps**: installed CUDA is **v13.2 — the exact version Unsloth flags as broken** (plus ancient v11.0). No `cmake`/`ninja` on PATH (VS BuildTools bundles cmake but not on PATH). VS 18 BuildTools (MSVC 14.50) present. `rg`, `git`, `hf` CLI present. Python torch is CPU-only (irrelevant — we don't use PyTorch).
- Sampling (Qwen official): thinking temp 1.0/top_p .95/top_k 20/min_p 0; coding-thinking temp 0.6; instruct temp 0.7/top_p .8/presence 1.5. Reasoning effort via `--chat-template-kwargs {"reasoning_effort":"xhigh|medium|low|none"}`.

## Design principles
1. **The KV cache is the product.** Every byte before the newest message must be identical to the previous request: system prompt + tools JSON frozen at session start (hashed), no timestamps, tool results append-only, compaction keeps `messages[0]` verbatim.
2. **One slot, one queue.** MTP forces `--parallel 1`; main agent, subagents, summarizer, commit-msg all serialize on one `EngineSlot` semaphore. Context switches mitigated with `--cache-ram` + `--slot-save-path`.
3. **Harness stays tiny** (< 150MB RSS): no in-process tokenizer by default (use `/tokenize` + blake3 cache + learned chars/token), redb mmap index, bounded transcript buffers, tool output spilled to disk.
4. **Small-model-proof tool protocol**: server-native parse → tolerant XML/fence parser → JSON repair → grammar-forced retry (`tool_choice: required`).

## Workspace layout

```
codeBuzz/
  Cargo.toml            # workspace, resolver 3; release: lto="fat", codegen-units=1, panic="abort", strip
  rust-toolchain.toml   # stable 1.97
  scripts/install-deps.ps1    # winget cmake+ninja; verify CUDA 13.1/13.3 present (not 13.2)
  scripts/build-llama.ps1     # clone/pin/cmake/ninja llama.cpp, CUDA sm_120 (also `buzzcode engine build`)
  eval/tasks/<name>/{repo/, task.md, check.ps1}
  crates/
    buzzcode/       # bin: clap CLI — `buzzcode`, `buzzcode -p "..." --json`, `engine {build|doctor|tune|status}`, `index`, `bench`
    cb-config/      # TOML schema + layering: ~/.buzzcode/config.toml < .buzzcode/config.toml < env < CLI
    cb-tool-api/    # tiny: Tool trait, ToolSpec, ToolOutput, ToolError, PermissionClass (avoids crate cycles)
    cb-engine/      # llama.cpp discover/build/download, GGUF header parser, VRAM planner, process supervisor, SSE client
    cb-core/        # messages, prefix-stable ContextStore, tokens, compaction, registry, permissions, parser, agent loop, subagents, plan mode
    cb-tools/       # read/write/edit/glob/grep/list_dir/shell/outline/symbol_search/repo_map/git_*
    cb-codeintel/   # tree-sitter symbols, redb index, PageRank repo map, AST chunking
    cb-mcp/         # MCP stdio client → Tool adapter
    cb-tui/         # ratatui UI
    cb-bench/       # tok/s, TTFT, cache-hit, eval runner (also a bin)
```
Deps flow: `buzzcode → cb-tui, cb-bench → cb-core → {cb-engine, cb-tools, cb-codeintel, cb-mcp} → cb-tool-api, cb-config`.

### Crates
| Concern | Crate | Why |
|---|---|---|
| async | `tokio` (rt-multi-thread, process, fs, sync, time, signal) | child supervision, SSE, TUI mux |
| HTTP/SSE | `reqwest` (json, stream, rustls-tls) + `eventsource-stream` | no OpenSSL on Windows |
| JSON | `serde`, `serde_json` **with `preserve_order`** | deterministic key order = stable prefix bytes |
| CLI/TUI | `clap` derive; `ratatui` + `crossterm` (event-stream) + `tui-textarea` | |
| parsing | `tree-sitter` 0.25 + grammar crates (rust, python, typescript, javascript, go, c, cpp, java, json, toml, md) | tags.scm per language via `include_str!` |
| tokens | llama-server `/tokenize` + `blake3` LRU; `tokenizers` behind off-by-default feature | keeps RSS low |
| search/walk | shell out to `rg --json` (installed); `ignore` crate for walking | zero RSS cost |
| diff | `similar` | edit previews, fuzzy-match hints |
| index | `redb` | pure Rust, mmap, ACID, no C build |
| fuzzy | `nucleo-matcher` | symbol search, tool-name repair |
| schema | `schemars` | tool arg JSON Schema from structs; reused for validation + grammar |
| MCP | `rmcp` (client, child-process transport) behind a `McpTransport` trait | hand-roll fallback ~400 LoC |
| misc | `tracing`+`tracing-appender` (file only), `anyhow`/`thiserror`, `toml`, `directories`, `indexmap`, `parking_lot`, `uuid`, `textwrap`, `unicode-width`, `rayon` | |
| GGUF / git / nvidia | hand-rolled GGUF header parser (~250 LoC); shell out to `git` and `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` | no libgit2/NVML builds |

## Engine (cb-engine)

Modules: `discover.rs`, `build.rs`, `download.rs` (HF Hub with Range resume + sha256), `gguf.rs`, `vram.rs`, `profile.rs`, `server.rs`, `client.rs`, `sse.rs`, `slot.rs`, `manager.rs`.

```rust
pub struct ModelFacts { arch: String, n_layer: u32, n_attn_layers: u32, n_recurrent_layers: u32,
    n_head_kv: u32, head_dim: u32, n_ctx_train: u32, has_mtp: bool,
    layer_bytes: Vec<u64>, nonlayer_bytes: u64, recurrent_state_bytes_per_seq: u64, chat_template: Option<String> }

pub struct VramPlanner { facts: ModelFacts, free_vram: u64, total_vram: u64, free_ram: u64 }
impl VramPlanner {
    fn kv_bytes_per_token(&self, kv: KvType) -> u64 { 2 * n_attn_layers * n_head_kv * head_dim * bytes_per_elem(kv) } // q8_0 ≈ 1.0625
    fn estimate(&self, ngl: u32, n_ctx: u32, o: &PlanOpts) -> u64 {
        weights(ngl) + kv_bytes_per_token*n_ctx + recurrent_state*(1+ctx_checkpoints)
        + compute(ubatch) /*~0.5GB @ ub512*/ + mtp /*~0.3GB*/ + CUDA_CTX /*0.6GB*/ + SAFETY /*0.35GB*/ }
    fn plan(&self, prefs: &PlanPrefs) -> VramPlan;  // see algorithm
}
pub struct VramPlan { ngl: u32, n_ctx: u32, override_tensor: Option<String>, est_vram: u64, est_ram: u64, ctx_checkpoints: u32, rationale: String }
```
`plan()`: (1) usable = free_vram − safety; (2) try all layers @ pref_ctx 32K; (3) else step ctx down to ctx_min 16K by 4K; (4) else offload **FFN of the last k blocks** via `-ot "blk\.(5[8-9]|6[0-3])\.ffn_.*=CPU"` (keeps attention+recurrent tensors on GPU; finer than `-ngl`), k↑ until fit; fallback to plain `-ngl N`; (5) persist to `~/.buzzcode/tune/<gguf-blake3>.toml`. Startup OOM (stderr `cudaMalloc`/`out of memory`) → `ngl -= 2` / `k += 1`, max 4 retries.

### VRAM budget (this machine: ~13.7GB free → ~13.3GB plannable)
| Option | Weights GPU | Fixed (CUDA+compute+MTP+rec) | KV q8 | Total | Fits? | Expected decode |
|---|---|---|---|---|---|---|
| UD-Q3_K_XL full GPU, 32K | 13.0 | 1.6 | 1.06 | 15.7 | No | — |
| Q3_K_S/M (~11.5) full GPU, 32K | 11.5 | 1.6 | 1.06 | 14.2 | Borderline | ~28 base / ~40 MTP |
| **IQ3_XXS/IQ3_M (~10.5) full GPU, 32K** | 10.5 | 1.6 | 1.06 | **13.2** | **Yes** | ~30 base / ~42 MTP |
| UD-Q3_K_XL, FFN last 12 blocks→CPU, 24K | 11.1 | 1.6 | 0.8 | 13.5 | Marginal | ~20 / ~28 |
| Q4_K_M (on disk), FFN last 22→CPU, 16K | 12.0 | 1.6 | 0.53 | 14.1 | Marginal | ~12–15 / ~18 |
| Q4_K_M, `-ngl 40` | 9.8 | 1.4 | 0.53 | 11.7 | Yes | ~8–10 (unusable) |

**Default profile:** `fast` = UD-IQ3_XXS (or Q3_K_S ≤ 11.5GB), 32K ctx, q8_0 KV, full GPU, MTP on. `quality` = UD-Q3_K_XL with planner-computed `-ot` offload, 24K ctx. The Q4_K_M already on disk is the offload *benchmark fixture*, not the daily driver. Bandwidth ceiling 448/11 ≈ 40 tok/s; ~70% eff ≈ 28; MTP n=2 @ ~60% acceptance ≈ 38–42 tok/s. Rule of thumb: every 1GB offloaded to CPU ≈ −35% decode. Closing Chrome/Animate before a session buys ~2GB → Q3_K_XL fits.

### Default llama-server command line
```
llama-server.exe -m <model.gguf> --alias qwen3.8-27b --host 127.0.0.1 --port 8089 --no-webui --metrics
  -ngl 99  [or planner: -ngl N / -ot "blk\.(5[8-9]|6[0-3])\.ffn_.*=CPU"]
  -c 32768 -fa on --cache-type-k q8_0 --cache-type-v q8_0
  -b 2048 -ub 512 -t 8 -tb 16 --no-mmap
  --jinja --reasoning-format deepseek
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
  --ctx-checkpoints 8 --checkpoint-min-step 2048
  --cache-ram 6144 --slot-save-path %USERPROFILE%\.buzzcode\slots
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
  --log-file %USERPROFILE%\.buzzcode\logs\llama-server.log --log-timestamps
```
Notes: `-t 8` = P-cores only (E-cores slow reductions); `--no-mmap` is the Windows knob (`--mlock` is a no-op); **never** `--cache-reuse` (hybrid model). `--spec-type ngram-simple --spec-draft-n-max 64` is an alternate `rewrite` profile benchmarked against MTP at M6 (requires restart → per-session choice). Per-request the harness sets `chat_template_kwargs.reasoning_effort`, sampling by task class, `id_slot: 0`, `cache_prompt: true`, `n_predict`, `tools`, `tool_choice`, `response_format`.

### Build script (`scripts/build-llama.ps1`)
```
winget install Kitware.CMake Ninja-build.Ninja
# Install CUDA 13.1 or 13.3+ side-by-side (13.2 flagged by Unsloth; 12.8 rejects MSVC 14.50 host compiler)
$env:CUDA_PATH = "...\CUDA\v13.1"
git clone https://github.com/ggml-org/llama.cpp B:\llama\llama.cpp ; git checkout <pin ≥ b10450 from config engine.pin>
# inside vcvars64 env from "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat":
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
  -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe" -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build build --config Release --target llama-server llama-bench -j 20
```
If nvcc rejects MSVC 14.50 → add `-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"`; record in `~/.buzzcode/engine/build-info.toml`. Fallback `engine.provider = "prebuilt"` downloads `llama-bXXXX-bin-win-cuda-13.x-x64.zip` + cudart from GitHub releases. `buzzcode engine doctor` = start server, `/health`, 64-token coherence check (the DeltaNet-kernel sanity test), `llama-bench -p 512 -n 128` recorded.

### EngineManager / client
```rust
pub struct EngineManager { cfg, profile, facts: ModelFacts, plan: VramPlan, proc: Mutex<Option<ServerProcess>>,
    client: LlamaClient, slot: Arc<EngineSlot>, state: watch::Sender<EngineState> }
pub enum EngineState { Stopped, Starting, Ready { n_ctx: u32 }, Crashed { attempts: u32 }, Restarting }
impl EngineManager {
    async fn ensure_binary(&self) -> Result<PathBuf>;   // discover → build / prebuilt
    async fn ensure_model(&self) -> Result<PathBuf>;    // local path (incl. LM Studio dirs) or HF download w/ resume
    async fn start(&self) -> Result<()>;                // plan → spawn → /health poll → warmup → slot save
    async fn warmup(&self, system_prefix: &str);        // system-only chat n_predict=1 → /slots/0?action=save&filename=warm.bin
    async fn restart_on_crash(self: Arc<Self>);         // backoff 1/2/4s, max 5
    async fn autotune(&self, sweep: SweepSpec) -> Result<TunedProfile>;  // M6
}
pub struct LlamaClient { http: reqwest::Client, base: Url, alias: String }
impl LlamaClient { chat_stream(ChatRequest) -> Stream<StreamEvent>; tokenize(&str)->u32; props(); health(); slot_save/restore(id,name); metrics() }
pub enum StreamEvent { Reasoning(String), Content(String), ToolCallDelta{index,id,name,args_fragment}, Usage{..},
    Timings{prompt_n, cache_n, prompt_ms, predicted_n, predicted_ms, draft_n, draft_accepted}, Done(FinishReason) }
pub struct EngineSlot { sem: Semaphore /*1*/, owner: Mutex<Option<ContextId>> }  // SlotGuard.switched=true → expect cache miss, log it
```

## Core (cb-core)

Modules: `message.rs`, `context.rs`, `tokens.rs`, `compaction.rs`, `prompt.rs`, `tool.rs`(registry), `permission.rs`, `parser.rs`, `agent.rs`, `guard.rs`, `task.rs`, `subagent.rs`, `plan.rs`, `events.rs`, `truncate.rs`.

### Prefix-stable ContextStore
```rust
pub enum Role { System, Developer, User, Assistant, Tool }
pub struct Message { role, content: String, tool_calls: Vec<ToolCall>, tool_call_id: Option<String>,
    tokens: Cell<Option<u32>>, meta: MsgMeta /* never serialized */ }
pub struct FrozenPrefix { system: Message, tools_json: Arc<Value>, bytes_hash: [u8;32], tokens: u32 }
pub struct ContextStore { id, prefix: FrozenPrefix, log: Vec<Message> /* append-only */, summary_boundary: usize, n_ctx: u32, counter: Arc<TokenCounter> }
impl ContextStore {
    fn push(&mut self, m: Message) -> &Message;                 // ONLY mutation path besides compact
    fn to_request_messages(&self) -> Vec<Value>;                // fixed key order: role, content, tool_calls, tool_call_id
    fn needs_compaction(&self, threshold: f32) -> bool;         // used > n_ctx*0.75 − 4096 reserve
    fn compact(&mut self, summary: String, keep_tail_turns: usize); // log = [User(summary)] ++ last N complete turns (no orphan tool results)
    fn fork_for_subagent(&self, kind: SubagentKind) -> ContextStore;
}
```
Rules (unit-tested by hashing consecutive serializations): tools array sorted by name, frozen once (`ToolRegistry::freeze()` → `Arc<Value>`; MCP tools must connect before freeze); repo map is **message #1** (Developer role if template supports it, else User with fixed header), refreshed only at compaction; assistant reasoning not stored; truncation before push; tool results never edited.

`TokenCounter`: blake3 → LRU → `/tokenize`; sync `estimate()` via learned chars/token; `calibrate()` from server `prompt_n + cache_n` after each request (ground truth).

### Tool API (cb-tool-api)
```rust
pub enum PermissionClass { ReadOnly, WriteFs, Exec, Network }
pub enum PermissionMode { Ask, AutoAcceptEdits, Yolo }
pub struct ToolSpec { name: &'static str, description: &'static str, schema: schemars::Schema, class: PermissionClass }
pub struct ToolCtx { cwd: PathBuf, session: Arc<Session>, cancel: CancellationToken, budget: OutputBudget }
pub struct ToolOutput { text: String, is_error: bool, truncated: Option<TruncInfo>, diff_preview: Option<String>, edited: Option<EditRecord> }
#[async_trait] pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;
    async fn preview(&self, args: &Value, cx: &ToolCtx) -> Result<Option<String>, ToolError> { Ok(None) } // diff before permission prompt
    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError>;
}
pub struct ToolRegistry { tools: BTreeMap<String, Arc<dyn Tool>>, frozen: OnceCell<Arc<Value>> }
impl ToolRegistry { register(); scoped(allow: &[&str]) -> ToolRegistry; freeze() -> Arc<Value>; get(); validate(&ToolCall) -> Result<(), ArgError> }
```
Each tool: `#[derive(Deserialize, JsonSchema)] struct Args` with `#[schemars(description)]` — schema feeds the model, validation, and grammar.

### ToolCallParser (layered)
```rust
pub enum ParseOutcome { Calls(ParsedCalls), NoCalls, Malformed { error: String, snippet: String } }
impl ToolCallParser { fn from_stream(&self, native: &[ToolCallDeltaAcc], content: &str) -> ParseOutcome
  // 1 server-native tool_calls (llama-server --jinja parses Qwen <tool_call>) → validate args
  // 2 regex <tool_call>\s*(\{.*?\})\s*</tool_call> (dotall, multi)
  // 3 ```json fences with {"name": <known>, "arguments"|"parameters": {...}}
  // 4 bare trailing JSON object with known "name"
  // 5 JSON repair (trailing commas, single quotes, raw newlines in strings, missing braces, True/None)
  // 6 fuzzy tool-name fix via nucleo
}
```
Retry escalation: Malformed → tool result with error + format reminder; 2nd → re-request with `tool_choice:"required"` (server GBNF from schemas) + `reasoning_effort:none`; 3rd → surface to user. `ArgError` → tool result with schema excerpt + field error.

### Agent state machine
```rust
pub enum AgentState { Idle, Preparing{turn}, Generating{turn, acc}, Parsing{turn}, AwaitingPermission{turn, pending},
    Executing{turn, running: JoinSet<..>}, Verifying{turn, edits}, Compacting, Finished(FinishReason) }
pub enum FinishReason { EndTurn, MaxTurns, UserAbort, LoopDetected, EngineError }
pub enum TaskClass { Chat, Plan, Explore, Edit, Summarize, CommitMsg }   // → sampling + reasoning_effort (plan=high, explore/chat=medium, edit=low, summarize/commit=none)
impl Agent {
    async fn run_turn(&mut self, input: String) -> Result<FinishReason> {
        self.ctx.push(Message::user(input));
        loop {
            if self.ctx.needs_compaction(0.75) { self.compact().await?; }
            let req = self.build_request();                       // prefix+log, frozen tools, sampling(task_class), reasoning_effort, id_slot 0, cache_prompt
            let _slot = self.engine.slot().acquire(self.ctx.id).await;
            let (assistant, timings) = self.stream_assistant(req).await?;  // emits Delta events + Metrics{cache_n/prompt_n, tok/s}
            match self.parser.from_stream(&assistant.native_calls, &assistant.content) {
                NoCalls => return Ok(EndTurn),
                Malformed{..} => { self.push_parse_error(); self.retry_mode.escalate(); continue; }
                Calls(pc) => {
                    self.guard.observe(&pc.calls)?;              // LoopDetected: 3 identical (name+args hash) in last 6, or same error ×3
                    let approved = self.perms.resolve(&pc.calls, ..).await?;  // diff previews; denied → tool result "denied by user"
                    let results = self.execute(approved).await;   // ReadOnly concurrently (JoinSet); WriteFs/Exec sequential; truncation applied
                    for r in results { self.ctx.push(Message::tool(r)); }
                    if any_edited { self.verify_edits().await; }  // re-read region ±15 lines + run [verify] cmds → push as tool results
                    self.guard.turn += 1; if self.guard.turn >= max_turns { return Ok(MaxTurns); }
                }
            }
        }
    }
}
```
**M0 check:** confirm via `/apply-template` that changing `reasoning_effort` doesn't alter the rendered system prefix; if it does, pin one effort per session.

**Compaction:** trigger at 75% − 4K reserve; summarize with `TaskClass::Summarize` (≤1200 tokens: goals, decisions, files touched + line ranges, open problems, next steps), keep last 2 complete turns, refresh repo map. Tool-result elision only happens at compaction time (never incrementally — any history rewrite is a cache break).

**Subagents:** `spawn_agent {kind: Explore|Plan|General, task}` tool; separate `ContextStore` with own frozen prefix; scoped registry (Explore/Plan = read-only tools; General = all except `spawn_agent`); serialized on the same slot; parent KV evicted to `--cache-ram` and restored (M0 verifies this for the hybrid arch; fallback = explicit `slot_save`/`slot_restore`, ~1–2s @ 32K). Result capped at 2K tokens, structured {findings, files, next}. **Secondary model slot** (`[[model_slot]]`) is designed but **disabled by default** — a 5–6GB GPU secondary would force the 27B down to IQ2 (unacceptable); option to run gemma-4-12B on CPU (`-ngl 0`, ~6–8 tok/s) for summaries/commit messages.

**Plan mode:** read-only registry + `write_plan` tool; plan JSON `{title, summary, steps:[{n, description, files, verification}], risks, questions}` validated by schemars; 2 failures → `tool_choice` forced to `write_plan` (grammar). Saved to `.buzzcode/plans/<slug>.{md,json}`; TUI approve/edit/reject → act session whose first user message is the plan (system prefix identical across plan/act).

## Tools (cb-tools)
| Tool | Class | Behaviour |
|---|---|---|
| `read_file {path, offset?, limit?}` | RO | line-numbered, default 400 lines/32KB, binary detection, total-lines hint |
| `write_file {path, content}` | WriteFs | preview = diff vs existing; creates dirs; refuses outside cwd unless yolo |
| `edit_file {path, old_string, new_string, replace_all?}` | WriteFs | exact match; 0 hits → nearest fuzzy match shown (`similar`); >1 hits → line list; unified-diff preview; emits `EditRecord` |
| `glob`, `list_dir` | RO | `ignore` walker, mtime-sorted, caps 500/300 |
| `grep {pattern, path?, glob?, context?, max_results?}` | RO | `rg --json`, grouped by file, cap 200 |
| `shell {command, timeout_s?, cwd?}` | Exec | `powershell -NoProfile -NonInteractive` (or `pwsh`); job object kills child tree on timeout; head 150/tail 50; denylist patterns always prompt |
| `outline {path}`, `symbol_search {query, kind?}`, `repo_map {focus_paths?, budget_tokens?}` | RO | tree-sitter index |
| `git_status`, `git_diff`, `git_log` / `git_commit {message?, add_all?}` | RO / WriteFs | commit msg generated via `TaskClass::CommitMsg` from staged diff (≤6K tokens); never `--no-verify` |
| `spawn_agent`, `write_plan`, `mcp__<server>__<tool>` | per kind / trust | |

Truncation (`truncate.rs`): `OutputBudget { max_bytes: 24_000, head_lines: 150, tail_lines: 50 }`; overflow → spill to `.buzzcode/tool-out/<turn>-<tool>-<n>.txt` + `[N lines omitted — read_file(path=…, offset=151)]`.
Edit-verify loop: after edits, auto-push re-read of each edited region, then run `[verify]` commands matching file glob (60s cap, last 80 lines), failures as `is_error=true`; at most once per 3 edits and always at end of turn.

## Code intelligence (cb-codeintel)
Modules: `lang.rs`, `parse.rs` (thread-local parser pool), `index.rs` (redb), `graph.rs` (PageRank), `repomap.rs`, `chunk.rs`, `outline.rs`.
```rust
pub struct Symbol { name: SmolStr, kind: SymKind, range: (u32,u32), signature: String /*≤120 chars*/, parent: Option<u32> }
// redb: FILES path→FileRec{mtime_ns,size,blake3,lang,symbols,ref_names}; DEFS name→Vec<(path_id,sym_idx)>; PATHS id↔path; META
impl Indexer { fn scan(&self, Incremental|Full) -> ScanStats /*rayon, skip unchanged mtime+size, batch txn 200*/; fn update_paths(&[PathBuf]) /*after edits*/; fn outline(); fn search(q, kind, limit) /*nucleo over in-memory DEFS keys*/ }
impl RepoMap { fn build(idx) /*edges F→D weight=ref count; PageRank d=.85, 30 iters; personalization: focus files + chat-mentioned identifiers ×10*/;
               fn render(idx, budget_tokens, focus, mentions, est) -> String /*greedy by rank, binary-search symbols-per-file to fit budget*/ }
```
Budget 2048 tokens (4096 in plan mode). Target: 5K-file repo full scan < 3s, index < 50MB. Languages v1: rust, python, ts/tsx, js, go, c, cpp, java, json, toml, markdown.

## MCP (cb-mcp)
`McpServerConfig {name, command, args, env, trust: readonly|ask|yolo, enabled}` → `McpClient` (rmcp child-process transport, initialize handshake) → `McpTool` implementing `Tool` (name `mcp__{server}__{tool}`, schema = inputSchema, class from trust, content[] → text, same truncation). Connect before registry freeze; crash → error + reconnect with backoff on next call.

## TUI (cb-tui)
Layout: transcript viewport / status line (`model · ctx 12.3k/32k · 41 tok/s · cache 97% · mode ASK · effort medium · turn 7/40`) / multi-line input. Modals: permission+diff (`y/n/a/v`), plan review, engine log (`F2`), help. `tokio::select!` over crossterm EventStream, `mpsc<AgentEvent>`, `watch<EngineState>`, 33ms dirty-only render. Reasoning dimmed/collapsible; tool calls as compact cards. Slash commands: `/plan /act /mode /effort /compact /map /index /engine /cost /clear`. Headless: `buzzcode -p "..." --json` (JSON-lines events; used by eval).

## System prompt (frozen, ~1.2–1.8K tokens)
Identity + contract → environment facts (OS, shell, cwd, git, top-3 languages) → tool protocol (native format, batch read-only calls, never fabricate output, you'll receive re-read + lint after edits) → `edit_file` rules (exact string, 3+ context lines, no whole-file rewrites) → output style → two ≤300-token worked examples → loop hygiene. Repo map lives in message #1, never in system bytes.

## Config schema (`~/.buzzcode/config.toml` ← `.buzzcode/config.toml`)
```toml
[general]  default_profile = "qwen38-27b-fast"; permission_mode = "ask"; max_turns = 40; log_level = "info"
[engine]   provider = "source"  # source | prebuilt | external
           llama_cpp_dir = "B:/llama/llama.cpp"; pin = "b10480"; cuda_path = ".../CUDA/v13.1"; cuda_arch = "120"
           host = "127.0.0.1"; port = 8089; restart_max = 5; cache_ram_mb = 6144; slot_save_dir = "~/.buzzcode/slots"
           models_dir = "B:/models"; extra_search_dirs = ["C:/Users/singh/.lmstudio/models"]
[[profile]] name = "qwen38-27b-fast"; model = "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-IQ3_XXS.gguf"; alias = "qwen3.8-27b"
           ctx = 32768; ctx_min = 16384; kv_type = "q8_0"; flash_attn = true; batch = 2048; ubatch = 512; threads = 8; threads_batch = 16
           ngl = "auto"; override_tensor = "auto"; spec = "draft-mtp"; spec_draft_n_max = 2; ctx_checkpoints = 8; checkpoint_min_step = 2048
           reasoning_format = "deepseek"; chat_template_file = ""; extra_args = []
  [profile.sampling.thinking] temperature=1.0; top_p=0.95; top_k=20; min_p=0.0
  [profile.sampling.coding]   temperature=0.6; top_p=0.95; top_k=20; min_p=0.0
  [profile.sampling.instruct] temperature=0.7; top_p=0.8; top_k=20; presence_penalty=1.5
  [profile.effort] plan="high"; explore="medium"; edit="low"; chat="medium"; summarize="none"; commit="none"
[[profile]] name = "qwen38-27b-quality"; model = "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q3_K_XL.gguf"; ctx = 24576
[[model_slot]] role = "secondary"; enabled = false; model = ".../gemma-4-12B-it-Q4_K_M.gguf"; device = "cpu"; port = 8090; parallel = 2; use_for = ["summarize","commit"]
[context]  compaction_threshold = 0.75; output_reserve_tokens = 4096; repo_map_tokens = 2048; tool_output_max_bytes = 24000; tool_output_head_lines = 150; tool_output_tail_lines = 50
[codeintel] index_path = ".buzzcode/index.redb"; languages = [...]; exclude = ["target/**","node_modules/**","dist/**"]
[verify]   "*.rs" = ["cargo check --message-format short"]; "*.py" = ["ruff check {file}"]; "*.ts" = ["npx tsc --noEmit -p ."]
[permissions] always_allow = [read-only tools]; always_ask_shell_patterns = ["rm -r","Remove-Item.*-Recurse","git push.*--force","git reset --hard"]
[[mcp_server]] name; command; args; trust = "readonly"; enabled = false
[bench]    eval_dir = "eval/tasks"
```

## Milestones
- **M0 Engine bring-up** — `cb-config`, `cb-engine` (gguf, vram, server, client, download), scripts, `buzzcode engine doctor` green. Verify with the real model: template renders tools + developer role; `reasoning_effort` vs prefix; LCP cache hit (`cache_n ≈ prompt_n − Δ`); `--cache-ram` restore on hybrid; slot save/restore; MTP acceptance; OOM → planner retry. Record in `docs/m0-results.md`.
- **M1 Agent loop (headless)** — `cb-tool-api`, `cb-core`, read/glob/grep/list_dir/shell; `buzzcode -p "what does this repo do" --json` works in yolo. Tests: prefix hash stability, 20-sample malformed parser corpus, truncation.
- **M2 Edits + TUI + permissions** — write/edit with previews, verify loop, full TUI, permission modal, real compaction, slash commands, crash-restart, RSS < 150MB @ 32K.
- **M3 Code intel** — indexer, redb, PageRank map ≤ 2K tokens, outline/symbol_search, index updated after edits.
- **M4 Subagents + plan mode + git** — spawn_agent with slot serialization + cache restore accounting; `write_plan` grammar fallback; git tools + generated commit messages; loop-guard tests.
- **M5 MCP** — stdio server tools appear namespaced with permission gating.
- **M6 Tuning + bench** — `buzzcode engine tune` sweeps ngl/`-ot`, ctx, `-t`, spec mode (draft-mtp vs ngram-simple vs none); `cb-bench` eval set of 10 tasks; defaults chosen from data.

## Verification
- `buzzcode engine doctor`: cmake/ninja on PATH; CUDA 13.1/13.3 nvcc + MSVC 14.50 compile; llama.cpp pin ≥ b10450; coherent 64-token output; GGUF parser reports n_layer=64 / 16 attn / has_mtp; planner estimate within ±10% of `nvidia-smi` after load.
- `buzzcode bench cache`: 10-turn scripted session → `cache_n/prompt_n ≥ 95%` after turn 1; prefix hash identical across turns; compaction logged as the only break.
- `buzzcode bench engine`: prompt tok/s @ 4K/16K/32K, decode tok/s, TTFT cold/warm, MTP acceptance, VRAM peak, harness RSS → markdown+JSON in `.buzzcode/bench/`.
- `buzzcode bench eval`: 10 fixture tasks (add fn+test rust, fix failing test py, rename across files ts, add CLI flag, docstring, grep-and-explain, multi-file refactor go, fix compile error, add TOML key, plan-only) → pass rate, turns, tokens, wall time; compare MTP vs ngram-simple vs none.
- Manual: `edit_file` rejects ambiguous matches and shows diff in Ask mode; lint output reaches the model and it self-corrects; Ctrl-C cancels stream and slot stays healthy; engine crash auto-restarts ≤5× and session continues; plan→approve→act keeps identical system-prefix hash; `git_commit` never passes `--no-verify`; MCP server death handled; `Get-Process buzzcode` RSS < 150MB.

## Risks
- Qwen3.8 chat template / tool parser lag in llama-server → `chat_template_file` override + our parser layers (test day one).
- `reasoning_effort` changes system rendering → pin per session.
- `--cache-ram` may not restore recurrent state → explicit slot save/restore around subagents.
- 8 checkpoints × recurrent state may cost 1–2GB → planner accounts; drop to 4 if tight.
- CUDA 13.1 + MSVC 14.50 → `-allow-unsupported-compiler` or `prebuilt` provider.
- `-ot` syntax drift across llama.cpp versions → validate at startup, fall back to `-ngl`.

## Critical files
- `crates/cb-engine/src/vram.rs` — GGUF-driven budget formula + ngl/ctx/`-ot` planner
- `crates/cb-engine/src/manager.rs` — process supervision, warm-up, restart, autotune entry
- `crates/cb-core/src/context.rs` — prefix-stable ContextStore, compaction, token accounting
- `crates/cb-core/src/agent.rs` — state machine, execution, verify loop, guards
- `crates/cb-core/src/parser.rs` — layered tool-call parser + repair + escalation
- `crates/cb-codeintel/src/repomap.rs` — PageRank symbol map under token budget
- `scripts/build-llama.ps1` — pinned CUDA sm_120 build
