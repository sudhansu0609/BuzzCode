//! `buzzcode engine ...` subcommands.

use anyhow::{Context, Result};
use cb_config::Config;
use cb_engine::client::ChatRequest;
use cb_engine::{discover, nvidia, pidfile, sentinel, EngineManager, EngineState, GgufFile};
use clap::Subcommand;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Subcommand, Debug)]
pub enum EngineCmd {
    /// Build llama.cpp from source with CUDA (runs scripts/build-llama.ps1).
    Build {
        #[arg(long)] pin: Option<String>,
        #[arg(long)] clean: bool,
        #[arg(long)] allow_cuda_132: bool,
        #[arg(long)] allow_unsupported_compiler: bool,
    },
    /// Download a prebuilt Windows CUDA release instead of building.
    Prebuilt { #[arg(long)] tag: Option<String> },
    /// Full health check: toolchain, binary, model, VRAM plan, server start, coherence, cache, tools, MTP.
    Doctor {
        /// Skip the live server tests (only static checks).
        #[arg(long)] r#static: bool,
        /// Keep the server running after doctor finishes.
        #[arg(long)] keep: bool,
    },
    /// Inspect a GGUF file (or the active profile's model).
    Inspect { path: Option<PathBuf>, #[arg(long)] tensors: bool },
    /// Show the VRAM plan for the active profile without starting anything.
    Plan,
    /// List GGUF models found in models_dir + extra_search_dirs.
    Models,
    /// Start the server in the foreground (Ctrl-C or `engine stop` stops it). Useful for external clients.
    Serve,
    /// Stop the engine recorded in ~/.buzzcode/engine.json (server tree + the serving process).
    Stop,
    /// Stop a running engine if there is one, then serve again.
    Restart,
    /// Status of a running engine (engine.json + health + /props + /slots).
    Status,
    /// Sweep speculative-decoding modes and record the fastest.
    Tune,
}

pub async fn run(cfg: Config, cmd: EngineCmd) -> Result<()> {
    let cfg = Arc::new(cfg);
    match cmd {
        EngineCmd::Build { pin, clean, allow_cuda_132, allow_unsupported_compiler } => {
            let opts = cb_engine::build::BuildOptions { pin, clean, allow_cuda_132, allow_unsupported_compiler };
            let exe = cb_engine::build::build_from_source(&cfg, &opts, |l| println!("{l}")).await?;
            println!("OK: {}", exe.display());
            Ok(())
        }
        EngineCmd::Prebuilt { tag } => {
            let exe = cb_engine::build::fetch_prebuilt(&cfg, tag.as_deref(), |l| println!("{l}")).await?;
            println!("OK: {}", exe.display());
            Ok(())
        }
        EngineCmd::Inspect { path, tensors } => inspect(&cfg, path, tensors),
        EngineCmd::Plan => plan(&cfg),
        EngineCmd::Models => {
            for (p, size) in discover::list_local_models(&cfg) {
                println!("{:>8.2} GiB  {}", size as f64 / (1u64 << 30) as f64, p.display());
            }
            Ok(())
        }
        EngineCmd::Status => status(&cfg).await,
        EngineCmd::Serve => serve(cfg).await,
        EngineCmd::Stop => stop(&cfg),
        EngineCmd::Restart => restart(cfg).await,
        EngineCmd::Doctor { r#static, keep } => doctor(cfg, r#static, keep).await,
        EngineCmd::Tune => crate::cmd_bench::tune((*cfg).clone()).await,
    }
}

/// Where the engine really is: `engine.json` first (written by the process that
/// is serving), then the shared port ledger, then the configured preference.
///
/// GUARDIAN_PLAN.md section 11 rule 5, minus the identity scan — `/health` on
/// llama-server does not name itself, so the two files are the only honest
/// sources and the config is the last resort.
fn live_base_url(cfg: &Config) -> String {
    if let Some(rec) = pidfile::read(cfg) {
        return format!("http://{}:{}", cfg.engine.host, rec.port);
    }
    if let Some(port) = cb_engine::ports::published_port(&cb_engine::ports::ledger_path(), cb_engine::ports::APP) {
        return format!("http://{}:{}", cfg.engine.host, port);
    }
    cfg.engine.base_url()
}

fn resolve_model(cfg: &Config, path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(p) => Ok(p),
        None => {
            let spec = &cfg.active_profile().model;
            discover::find_model(cfg, spec).with_context(|| format!("model {spec:?} not found locally; run `buzzcode engine doctor` to download"))
        }
    }
}

fn inspect(cfg: &Config, path: Option<PathBuf>, tensors: bool) -> Result<()> {
    let p = resolve_model(cfg, path)?;
    let g = GgufFile::open(&discover::first_shard(&p))?;
    let f = g.facts();
    println!("{}", f.summary());
    println!("  file size       : {:.2} GiB", f.file_size as f64 / (1u64 << 30) as f64);
    println!("  non-layer bytes : {:.2} GiB", f.nonlayer_bytes as f64 / (1u64 << 30) as f64);
    if !f.layer_bytes.is_empty() {
        let avg = f.layer_bytes.iter().sum::<u64>() as f64 / f.layer_bytes.len() as f64;
        println!("  avg layer bytes : {:.1} MiB (ffn share {:.0}%)", avg / (1u64 << 20) as f64,
            100.0 * f.layer_ffn_bytes.iter().sum::<u64>() as f64 / f.layer_bytes.iter().sum::<u64>().max(1) as f64);
    }
    println!("  recurrent state : {:.0} MiB / seq", f.recurrent_state_bytes_per_seq as f64 / (1u64 << 20) as f64);
    println!("  mtp bytes       : {:.0} MiB", f.mtp_bytes as f64 / (1u64 << 20) as f64);
    println!("  chat template   : {} chars", f.chat_template.as_ref().map(|s| s.len()).unwrap_or(0));
    let mut keys: Vec<_> = g.metadata.iter().filter(|(k, _)| !k.starts_with("tokenizer.ggml.")).collect();
    keys.sort_by(|a, b| a.0.cmp(b.0));
    println!("  metadata:");
    for (k, v) in keys {
        let s = match v {
            cb_engine::gguf::MetaValue::Str(s) => { let t: String = s.chars().take(80).collect(); format!("{t:?}{}", if s.len() > 80 { "…" } else { "" }) }
            cb_engine::gguf::MetaValue::Array(a) => format!("[array len {}]", a.len()),
            other => format!("{other:?}"),
        };
        println!("    {k} = {s}");
    }
    if tensors {
        println!("  tensors ({}):", g.tensors.len());
        for t in &g.tensors {
            println!("    {:<40} {:<8} {:?} {:.1} MiB", t.name, cb_engine::gguf::ggml_type_name(t.ggml_type), t.dims, t.n_bytes() as f64 / (1u64 << 20) as f64);
        }
    }
    Ok(())
}

fn plan(cfg: &Arc<Config>) -> Result<()> {
    let profile = cfg.active_profile().clone();
    if profile.is_external(&cfg.engine) {
        println!("External Engine : {}", profile.base_url(&cfg.engine));
        println!("Model           : {}", profile.request_model());
        println!("Context Size    : {}", profile.ctx);
        println!("Note            : Managed externally at {}", profile.base_url(&cfg.engine));
        return Ok(());
    }
    let p = resolve_model(cfg, None)?;
    let g = GgufFile::open(&discover::first_shard(&p))?;
    let facts = g.facts();
    let gpu = nvidia::query()?;
    let mgr = EngineManager::new(cfg.clone(), profile.clone());
    let plan = mgr.compute_plan(&facts, gpu.mem.free_bytes(), gpu.mem.total_bytes());
    println!("GPU   : {} ({} MiB total, {} MiB used, {} MiB free)", gpu.name, gpu.mem.total_mib, gpu.mem.used_mib, gpu.mem.free_mib);
    println!("Model : {}", facts.summary());
    println!("Plan  : ngl={} ctx={} ffn_cpu_blocks={} ot={:?}", plan.ngl, plan.n_ctx, plan.ffn_cpu_blocks, plan.override_tensor);
    println!("        est VRAM {:.2} GiB, CPU weights {:.2} GiB", plan.est_vram_gib(), plan.est_cpu_weight_bytes as f64 / (1u64 << 30) as f64);
    let gpu_bytes = facts.weights_bytes_on_gpu(plan.ngl).saturating_sub(facts.ffn_tail_bytes(plan.ffn_cpu_blocks as usize));
    let tps = cb_engine::vram::estimate_decode_tps(gpu_bytes, plan.est_cpu_weight_bytes, 448.0, 60.0, 1.0);
    println!("        rough decode estimate: {:.0} tok/s base, ~{:.0} with MTP", tps, tps * 1.4);
    println!("--- rationale ---\n{}", plan.rationale);
    // The preferred port, because nothing is being started here. At start() the
    // engine steps forward if it is busy, so the live command line may well
    // carry a different --port.
    let args = cb_engine::server::LlamaServerArgs::build(cfg, &profile, PathBuf::from("llama-server.exe"), p, &plan, cfg.engine.port);
    println!("--- command ---\n{}", args.command_line());
    Ok(())
}

async fn status(cfg: &Config) -> Result<()> {
    // The record first: it is the only thing that answers "who is serving, and
    // can I stop it?" when the server has wedged and `/health` never returns.
    let path = pidfile::path(cfg);
    match pidfile::read(cfg) {
        None => println!("engine.json : none at {}", path.display()),
        Some(rec) => {
            let serving = pidfile::alive(rec.pid);
            let server_alive = rec.server_pid.map(pidfile::alive).unwrap_or(false);
            println!("engine.json : {}", path.display());
            println!("  pid       : {} ({})", rec.pid, if serving { "running" } else { "GONE — stale record" });
            match rec.server_pid {
                Some(sp) => println!("  server_pid: {sp} ({})", if server_alive { "running" } else { "GONE" }),
                None => println!("  server_pid: -"),
            }
            println!("  port      : {}", rec.port);
            println!("  model     : {}", rec.model);
            println!("  alias     : {}", rec.alias);
            println!("  started   : {}", rec.started_at);
            if !serving && !server_alive { println!("  → `buzzcode engine stop` clears the stale record."); }
        }
    }

    // The port the engine actually took, in the order the plan gives: what
    // `engine serve` recorded, then the shared ledger, then the configured
    // wish. Asking the config alone is how `engine status` used to report a
    // healthy engine as DOWN after it had stepped forward off a busy 8089.
    let client = cb_engine::LlamaClient::with_api_key(&live_base_url(cfg), cfg.engine.resolved_api_key());
    let healthy = client.health().await?;
    println!("engine {} : {}", client.base_url(), if healthy { "healthy" } else { "DOWN" });
    if healthy {
        if let Ok(props) = client.props().await {
            if !props.model_path.is_empty() { println!("  model   : {}", props.model_path); }
            if props.n_ctx() > 0 { println!("  n_ctx   : {}", props.n_ctx()); }
            if props.total_slots > 0 { println!("  slots   : {}", props.total_slots); }
            if !props.build_info.is_empty() { println!("  build   : {}", props.build_info); }
        }
        if let Ok(s) = client.slots().await { println!("  /slots  : {}", serde_json::to_string(&s)?.chars().take(400).collect::<String>()); }
    }
    Ok(())
}

async fn serve(cfg: Arc<Config>) -> Result<()> {
    let profile = cfg.active_profile().clone();
    let mgr = EngineManager::new(cfg.clone(), profile);
    let mut rx = mgr.watch_state();
    let m2 = mgr.clone();
    tokio::spawn(async move {
        while rx.changed().await.is_ok() { println!("[engine] {:?}", rx.borrow().clone()); }
        let _ = m2;
    });

    // Listen for Sentinel before we take the card, so an order to give it back
    // that arrives during a slow model load is not lost.
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let sentinel = sentinel::Sentinel::new();
    let me = sentinel.client().to_string();
    let _events = sentinel.subscribe(move |ev| {
        if let Some(reason) = sentinel::stop_reason(&ev, &me) { let _ = stop_tx.send(reason); }
    });

    // Registers and reserves with Sentinel; exits 3 through main() if refused.
    mgr.start().await?;

    let model = mgr.model_path().map(|p| cb_engine::server::model_display(&p)).unwrap_or_default();
    // The real port, not the configured one: `engine status` and `engine stop`
    // read this file, and the harness's own client uses mgr.base_url().
    let rec = cb_engine::EngineRecord::new(mgr.port(), model, mgr.alias(), mgr.server_pid());
    if let Err(e) = pidfile::write(&cfg, &rec) {
        // Not fatal: the engine is up, we just cannot be stopped by name.
        eprintln!("warning: could not write {}: {e:#}", pidfile::path(&cfg).display());
    }
    println!("serving at {} — Ctrl-C or `buzzcode engine stop` to stop", mgr.base_url());
    if mgr.port() != cfg.engine.port {
        println!("  (configured port {} was busy; stepped forward)", cfg.engine.port);
    }
    println!("  pid {} · server pid {} · {}", rec.pid, rec.server_pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into()), pidfile::path(&cfg).display());

    tokio::select! {
        r = tokio::signal::ctrl_c() => { r?; println!("\nstopping…"); }
        Some(reason) = stop_rx.recv() => { println!("\n{reason}"); }
    }
    mgr.stop().await;
    pidfile::remove(&cfg);
    Ok(())
}

/// Terminate whatever `engine.json` names. Returns what was actually killed.
///
/// The server tree goes first: killing the supervising `buzzcode` first would
/// orphan `llama-server` with all of its VRAM still held.
fn stop_recorded(cfg: &Config) -> Option<(cb_engine::EngineRecord, Vec<String>)> {
    let rec = pidfile::read(cfg)?;
    let mut killed = Vec::new();
    if let Some(sp) = rec.server_pid {
        if pidfile::alive(sp) && pidfile::kill_tree(sp) { killed.push(format!("llama-server (pid {sp})")); }
    }
    if rec.pid != std::process::id() && pidfile::alive(rec.pid) && pidfile::kill_tree(rec.pid) {
        killed.push(format!("buzzcode engine serve (pid {})", rec.pid));
    }
    pidfile::remove(cfg);
    Some((rec, killed))
}

fn stop(cfg: &Config) -> Result<()> {
    let Some((rec, killed)) = stop_recorded(cfg) else {
        eprintln!("no engine recorded in {} — nothing to stop", pidfile::path(cfg).display());
        std::process::exit(1);
    };
    if killed.is_empty() {
        eprintln!(
            "engine.json named pid {} (server {}), but neither is running — removed the stale record",
            rec.pid,
            rec.server_pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into())
        );
        std::process::exit(1);
    }
    println!("stopped {}", killed.join(" and "));
    Ok(())
}

async fn restart(cfg: Arc<Config>) -> Result<()> {
    match stop_recorded(&cfg) {
        Some((_, killed)) if !killed.is_empty() => {
            println!("stopped {}", killed.join(" and "));
            // Give Windows a moment to release the listening port.
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        }
        Some(_) => println!("no live engine behind {} — starting a new one", pidfile::path(&cfg).display()),
        None => println!("no engine was running — starting one"),
    }
    serve(cfg).await
}

// ---------------------------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------------------------

struct Report { ok: u32, warn: u32, fail: u32 }
impl Report {
    fn ok(&mut self, m: impl AsRef<str>) { self.ok += 1; println!("  [ok]   {}", m.as_ref()); }
    fn warn(&mut self, m: impl AsRef<str>) { self.warn += 1; println!("  [warn] {}", m.as_ref()); }
    fn fail(&mut self, m: impl AsRef<str>) { self.fail += 1; println!("  [FAIL] {}", m.as_ref()); }
}

fn which(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) && !name.ends_with(".exe") { format!("{name}.exe") } else { name.to_string() };
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join(&exe)).find(|p| p.is_file()))
}

async fn doctor(cfg: Arc<Config>, static_only: bool, keep: bool) -> Result<()> {
    let mut r = Report { ok: 0, warn: 0, fail: 0 };
    println!("== buzzcode engine doctor ==");

    // --- toolchain ---
    println!("toolchain:");
    for t in ["cmake", "ninja", "git", "rg"] {
        match which(t) { Some(p) => r.ok(format!("{t:<6} {}", p.display())), None => r.warn(format!("{t} not on PATH")) }
    }
    let cuda_root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if cuda_root.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(&cuda_root)?.flatten().map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.starts_with('v')).collect();
        names.sort();
        let has_13_2 = names.iter().any(|n| n == "v13.2");
        let good: Vec<&String> = names.iter().filter(|n| { let v = n.trim_start_matches('v'); v != "13.2" && v.split('.').next().and_then(|m| m.parse::<u32>().ok()).unwrap_or(0) >= 12 && v != "12.0" && v != "12.1" && v != "12.2" && v != "12.3" && v != "12.4" && v != "12.5" && v != "12.6" && v != "12.7" }).collect();
        if good.is_empty() {
            if has_13_2 { r.warn("only CUDA 13.2 present — flagged by Unsloth for Qwen3.x; install 13.1 or 13.3+ (build can proceed with --allow-cuda-132; the coherence test below validates)"); }
            else { r.fail(format!("no CUDA >= 12.8 toolkit found (present: {})", names.join(", "))); }
        } else { r.ok(format!("CUDA toolkits usable: {}", good.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))); }
    } else { r.warn("CUDA toolkit directory not found (needed only to build from source)"); }

    // --- GPU ---
    println!("gpu:");
    let gpu = match nvidia::query() { Ok(g) => { r.ok(format!("{} driver {} — {} MiB total, {} MiB free", g.name, g.driver, g.mem.total_mib, g.mem.free_mib)); Some(g) } Err(e) => { r.fail(format!("nvidia-smi: {e:#}")); None } };
    let (ram_t, ram_f) = nvidia::system_ram();
    r.ok(format!("RAM {:.1} GiB total, {:.1} GiB free", ram_t as f64 / (1u64 << 30) as f64, ram_f as f64 / (1u64 << 30) as f64));

    let profile = cfg.active_profile().clone();
    let is_external = profile.is_external(&cfg.engine);

    // --- binary ---
    println!("engine binary:");
    let binary = if is_external {
        r.ok(format!("external provider (base_url: {})", profile.base_url(&cfg.engine)));
        None
    } else {
        match discover::find_server(&cfg) {
            Ok(b) => {
                let ver = std::process::Command::new(&b).arg("--version").output().ok().map(|o| String::from_utf8_lossy(&o.stderr).to_string() + &String::from_utf8_lossy(&o.stdout)).unwrap_or_default();
                let ver_line = ver.lines().find(|l| l.contains("version") || l.contains("build")).unwrap_or("").trim().to_string();
                r.ok(format!("{} {}", b.display(), ver_line));
                if let Some(n) = ver_line.split_whitespace().find_map(|w| w.strip_prefix('b').or(Some(w)).and_then(|x| x.parse::<u32>().ok()).filter(|n| *n > 5000 && *n < 100000)) {
                    if n < 10450 { r.fail(format!("llama.cpp build b{n} is older than b10450; Qwen3.8 DeltaNet CUDA kernels were broken before that")); }
                }
                Some(b)
            }
            Err(e) => { r.fail(format!("{e:#}")); None }
        }
    };

    // --- model ---
    println!("model:");
    let model = if is_external {
        r.ok(format!("profile {:?} → external model {:?}", profile.name, profile.request_model()));
        None
    } else {
        match discover::find_model(&cfg, &profile.model) {
            Some(p) => { r.ok(format!("profile {:?} → {}", profile.name, p.display())); Some(discover::first_shard(&p)) }
            None => { r.warn(format!("model {:?} not present locally; `buzzcode engine doctor` without --static will download it", profile.model)); None }
        }
    };
    let mut facts = None;
    if let Some(m) = &model {
        match GgufFile::open(m) {
            Ok(g) => {
                let f = g.facts();
                r.ok(f.summary());
                if f.is_hybrid() { r.ok(format!("hybrid attention: {} full-attn + {} recurrent layers → small KV cache; --cache-reuse is NOT used", f.n_attn_layers, f.n_recurrent_layers)); }
                if profile.spec == "draft-mtp" && !f.has_mtp { r.warn("profile requests draft-mtp but GGUF has no MTP tensors — server will fall back / error; set spec = \"none\" or \"ngram-simple\""); }
                if f.has_mtp { r.ok("MTP head present (speculative decoding available)"); }
                if f.chat_template.is_none() { r.warn("no chat template embedded; set profile.chat_template_file"); }
                facts = Some(f);
            }
            Err(e) => r.fail(format!("GGUF parse: {e:#}")),
        }
    }

    // --- plan ---
    println!("vram plan:");
    let mgr = EngineManager::new(cfg.clone(), profile.clone());
    if is_external {
        r.ok(format!("managed by external server at {}", profile.base_url(&cfg.engine)));
    } else if let (Some(f), Some(g)) = (&facts, &gpu) {
        let plan = mgr.compute_plan(f, g.mem.free_bytes(), g.mem.total_bytes());
        r.ok(format!("ngl={} ctx={} ffn_cpu_blocks={} → est {:.2} GiB of {:.2} GiB free", plan.ngl, plan.n_ctx, plan.ffn_cpu_blocks, plan.est_vram_gib(), g.mem.free_bytes() as f64 / (1u64 << 30) as f64));
        if plan.ffn_cpu_blocks > 0 || (plan.ngl as usize) <= f.layer_bytes.len() { r.warn("model does not fully fit: expect slower decode. Close GPU-heavy apps (browsers, Adobe) or pick a smaller quant (IQ3_XXS / Q3_K_S)"); }
        if g.mem.used_mib > 1500 { r.warn(format!("{} MiB VRAM already in use by other processes", g.mem.used_mib)); }
    }

    if static_only || (!is_external && binary.is_none()) || (!is_external && gpu.is_none()) {
        return finish(r);
    }

    // --- live ---
    println!("live server:");
    let t0 = Instant::now();
    if let Err(e) = mgr.start().await { r.fail(format!("start: {e:#}")); return finish(r); }
    let state = mgr.state();
    match &state { EngineState::Ready { n_ctx, model } => r.ok(format!("ready in {:.1}s — {model} n_ctx={n_ctx}", t0.elapsed().as_secs_f64())), s => r.warn(format!("state {s:?}")) }
    if !is_external {
        if let (Some(g0), Ok(g1)) = (&gpu, nvidia::query()) {
            let used = g1.mem.used_mib.saturating_sub(g0.mem.used_mib);
            let est = mgr.plan().map(|p| p.est_vram_bytes >> 20).unwrap_or(0);
            let err = if est > 0 { (used as f64 - est as f64) / est as f64 * 100.0 } else { 0.0 };
            if err.abs() <= 12.0 { r.ok(format!("VRAM used by server {used} MiB vs planned {est} MiB ({err:+.0}%)")); }
            else { r.warn(format!("VRAM used by server {used} MiB vs planned {est} MiB ({err:+.0}%) — planner constants need adjusting")); }
        }
    }
    let client = mgr.client().clone();
    let alias = mgr.alias().to_string();

    // coherence
    let sys = json!({"role":"system","content":"You are a concise assistant."});
    let req = ChatRequest::new(&alias, vec![sys.clone(), json!({"role":"user","content":"Write one sentence about Rust, then list three prime numbers."})]);
    let req = ChatRequest { max_tokens: Some(256), ..req }.reasoning_effort("none");
    match client.chat_once(req).await {
        Ok((content, reasoning, t, _)) => {
            let text = if content.trim().is_empty() && !reasoning.trim().is_empty() { reasoning.as_str() } else { content.as_str() };
            let printable = text.chars().filter(|c| c.is_ascii_graphic() || c.is_whitespace()).count() as f64 / text.chars().count().max(1) as f64;
            let has_digit = text.chars().any(|c| c.is_ascii_digit());
            if printable > 0.90 && has_digit && text.len() > 15 { r.ok(format!("coherence: {:?}", text.trim().chars().take(100).collect::<String>())); }
            else { r.fail(format!("coherence test looks garbled: {:?}", text.chars().take(120).collect::<String>())); }
            r.ok(format!("decode {:.1} tok/s, prompt {:.0} tok/s{}", t.decode_tps(), t.prompt_tps(), t.draft_acceptance().map(|a| format!(", MTP acceptance {:.0}%", a * 100.0)).unwrap_or_default()));
        }
        Err(e) => r.fail(format!("chat: {e:#}")),
    }

    // prefix cache: same prefix, new user turn → cache_n should cover the prefix
    if !is_external {
        let long_sys = json!({"role":"system","content": format!("You are buzzcode, a coding agent. Rules:\n{}", (1..=60).map(|i| format!("{i}. Always read before editing; keep edits minimal; verify with tests.")).collect::<Vec<_>>().join("\n"))});
        let mk = |u: &str| ChatRequest { max_tokens: Some(8), ..ChatRequest::new(&alias, vec![long_sys.clone(), json!({"role":"user","content":u})]) }.reasoning_effort("none");
        let first = client.chat_once(mk("Say hi.")).await;
        let second = client.chat_once(mk("Say hello.")).await;
        match (first, second) {
            (Ok((_, _, t1, _)), Ok((_, _, t2, _))) => {
                let ratio = t2.cache_ratio();
                if ratio >= 0.85 { r.ok(format!("prefix cache reuse: turn1 prompt_n={} → turn2 cache_n={} prompt_n={} ({:.0}% cached, prompt {:.0} ms)", t1.prompt_n + t1.cache_n, t2.cache_n, t2.prompt_n, ratio * 100.0, t2.prompt_ms)); }
                else { r.fail(format!("prefix cache NOT reused: turn2 cache_n={} prompt_n={} ({:.0}%)", t2.cache_n, t2.prompt_n, ratio * 100.0)); }
            }
            (a, b) => r.fail(format!("cache test: {:?} / {:?}", a.err(), b.err())),
        }
    } else {
        r.ok("prefix cache test skipped for external provider (managed server-side)");
    }

    // reasoning_effort prefix stability
    let msgs = vec![sys.clone(), json!({"role":"user","content":"ping"})];
    let a = client.apply_template(&msgs, None, Some(&json!({"reasoning_effort":"high"}))).await;
    let b = client.apply_template(&msgs, None, Some(&json!({"reasoning_effort":"low"}))).await;
    match (a, b) {
        (Ok(a), Ok(b)) => {
            let common = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            let sys_end = a.find("ping").unwrap_or(0);
            if a == b { r.ok("reasoning_effort does not change the rendered prompt (template ignores it or handles it server-side)"); }
            else if common >= sys_end { r.ok(format!("reasoning_effort only changes the prompt tail (common prefix {common} chars ≥ system end {sys_end}) → safe to vary per turn")); }
            else { r.warn(format!("reasoning_effort changes the prompt BEFORE the user message (common prefix {common} < {sys_end}) → pin one effort per session")); }
        }
        (a, b) => r.warn(format!("/apply-template unavailable: {:?} {:?}", a.err(), b.err())),
    }

    // native tool calling
    let tools = json!([{"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"}},"required":["path"]}}}]);
    let mut req = ChatRequest::new(&alias, vec![json!({"role":"system","content":"You are a coding agent. Use tools when needed."}), json!({"role":"user","content":"Read the file src/main.rs"})]);
    req.tools = Some(tools.clone());
    req.max_tokens = Some(200);
    let req = req.reasoning_effort("none");
    match client.chat_once(req).await {
        Ok((content, _, _, calls)) => {
            if let Some(c) = calls.first() {
                let args_ok = serde_json::from_str::<Value>(&c.arguments).map(|v| v.get("path").is_some()).unwrap_or(false);
                if c.name == "read_file" && args_ok { r.ok(format!("native tool call parsed by server: {}({})", c.name, c.arguments)); }
                else { r.warn(format!("tool call present but odd: name={} args={}", c.name, c.arguments)); }
            } else if content.contains("<tool_call>") || content.contains("read_file") {
                r.warn(format!("server did not parse the tool call natively (harness parser will handle): {:?}", content.chars().take(160).collect::<String>()));
            } else { r.warn(format!("model did not call the tool: {:?}", content.chars().take(160).collect::<String>())); }
        }
        Err(e) => r.fail(format!("tool-call test: {e:#}")),
    }

    // slot save/restore
    if !is_external {
        match client.slot_save(0, "doctor-test.bin").await {
            Ok(v) => {
                let n = v.get("n_saved").and_then(Value::as_u64).unwrap_or(0);
                match client.slot_restore(0, "doctor-test.bin").await {
                    Ok(_) => r.ok(format!("slot save/restore works ({n} tokens)")),
                    Err(e) => r.warn(format!("slot restore failed: {e:#}")),
                }
                let _ = std::fs::remove_file(cfg.paths.slots_dir().join("doctor-test.bin"));
            }
            Err(e) => r.warn(format!("slot save unavailable: {e:#}")),
        }
    }

    if !keep { mgr.stop().await; } else { println!("server kept running at {}", mgr.base_url()); }
    finish(r)
}

fn finish(r: Report) -> Result<()> {
    println!("== {} ok, {} warnings, {} failures ==", r.ok, r.warn, r.fail);
    if r.fail > 0 { std::process::exit(1); }
    Ok(())
}
