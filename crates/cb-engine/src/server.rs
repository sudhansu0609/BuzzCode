//! llama-server argument construction + child process supervision.

use crate::vram::{KvType, VramPlan};
use anyhow::{bail, Context, Result};
use cb_config::{Config, Profile};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

#[derive(Debug, Clone)]
pub struct LlamaServerArgs {
    pub binary: PathBuf,
    pub model: PathBuf,
    pub args: Vec<String>,
}

impl LlamaServerArgs {
    /// `port` is the port the server will actually bind, which is not always
    /// `cfg.engine.port` — see `ports::pick_port` and GUARDIAN_PLAN.md section
    /// 11. Passing it explicitly is what keeps the spawned command line and the
    /// client's base URL from drifting apart.
    pub fn build(cfg: &Config, profile: &Profile, binary: PathBuf, model: PathBuf, plan: &VramPlan, port: u16) -> Self {
        let e = &cfg.engine;
        let kv = KvType::parse(&profile.kv_type);
        let mut a: Vec<String> = Vec::new();
        let push = |a: &mut Vec<String>, s: &str| a.push(s.to_string());

        a.push("-m".into()); a.push(model.to_string_lossy().into_owned());
        a.push("--alias".into()); a.push(profile.alias.clone());
        a.push("--host".into()); a.push(e.host.clone());
        a.push("--port".into()); a.push(port.to_string());
        push(&mut a, "--no-webui");
        push(&mut a, "--metrics");
        push(&mut a, "--slots");

        a.push("-ngl".into()); a.push(plan.ngl.to_string());
        if let Some(ot) = &plan.override_tensor {
            a.push("-ot".into()); a.push(ot.clone());
        } else if !profile.override_tensor.is_empty() && !profile.override_tensor.eq_ignore_ascii_case("auto") {
            a.push("-ot".into()); a.push(profile.override_tensor.clone());
        }
        a.push("-c".into()); a.push(plan.n_ctx.to_string());
        if profile.flash_attn {
            push(&mut a, "-fa"); push(&mut a, "on");
            a.push("--cache-type-k".into()); a.push(kv.flag().into());
            a.push("--cache-type-v".into()); a.push(kv.flag().into());
        }
        a.push("-b".into()); a.push(profile.batch.to_string());
        a.push("-ub".into()); a.push(profile.ubatch.to_string());
        a.push("-t".into()); a.push(profile.threads.to_string());
        a.push("-tb".into()); a.push(profile.threads_batch.to_string());
        if profile.no_mmap { push(&mut a, "--no-mmap"); }
        push(&mut a, "--jinja");
        if !profile.reasoning_format.is_empty() && !profile.reasoning_format.eq_ignore_ascii_case("none") {
            a.push("--reasoning-format".into()); a.push(profile.reasoning_format.clone());
        }
        if !profile.draft_model.is_empty() {
            let draft_path = crate::discover::find_model(cfg, &profile.draft_model)
                .unwrap_or_else(|| PathBuf::from(&profile.draft_model));
            a.push("--model-draft".into());
            a.push(draft_path.to_string_lossy().into_owned());
        }
        if !profile.chat_template_file.is_empty() {
            a.push("--chat-template-file".into()); a.push(cb_config::expand_path(&profile.chat_template_file).to_string_lossy().into_owned());
        }
        match profile.spec.as_str() {
            "none" | "" => { push(&mut a, "--parallel"); push(&mut a, "1"); }
            spec => {
                a.push("--spec-type".into()); a.push(spec.to_string());
                a.push("--spec-draft-n-max".into()); a.push(profile.spec_draft_n_max.to_string());
                push(&mut a, "--parallel"); push(&mut a, "1");
            }
        }
        if plan.ctx_checkpoints > 0 {
            a.push("--ctx-checkpoints".into()); a.push(plan.ctx_checkpoints.to_string());
            a.push("--checkpoint-min-step".into()); a.push(profile.checkpoint_min_step.to_string());
        }
        if e.cache_ram_mb > 0 { a.push("--cache-ram".into()); a.push(e.cache_ram_mb.to_string()); }
        a.push("--slot-save-path".into()); a.push(cfg.paths.slots_dir().to_string_lossy().into_owned());
        // Server-side sampling defaults (per-request values override these).
        let s = profile.sampling.coding;
        a.push("--temp".into()); a.push(s.temperature.to_string());
        a.push("--top-p".into()); a.push(s.top_p.to_string());
        a.push("--top-k".into()); a.push(s.top_k.to_string());
        a.push("--min-p".into()); a.push(s.min_p.to_string());
        a.push("--log-file".into()); a.push(cfg.paths.logs_dir().join("llama-server.log").to_string_lossy().into_owned());
        push(&mut a, "--log-timestamps");
        a.extend(profile.extra_args.iter().cloned());
        Self { binary, model, args: a }
    }

    pub fn command_line(&self) -> String {
        let mut s = format!("\"{}\"", self.binary.display());
        for a in &self.args {
            if a.contains(' ') || a.contains('|') || a.contains('\\') { s.push_str(&format!(" \"{a}\"")); } else { s.push(' '); s.push_str(a); }
        }
        s
    }
}

/// Ring buffer of recent stderr/stdout lines from the server (for the TUI log view and OOM detection).
#[derive(Default)]
pub struct LogTail {
    lines: Mutex<VecDeque<String>>,
    cap: usize,
}

impl LogTail {
    pub fn new(cap: usize) -> Self { Self { lines: Mutex::new(VecDeque::with_capacity(cap)), cap } }
    pub fn push(&self, line: String) {
        let mut l = self.lines.lock();
        if l.len() >= self.cap { l.pop_front(); }
        l.push_back(line);
    }
    pub fn snapshot(&self) -> Vec<String> { self.lines.lock().iter().cloned().collect() }
    pub fn contains(&self, needle: &str) -> bool { self.lines.lock().iter().any(|l| l.contains(needle)) }
}

pub struct ServerProcess {
    pub child: Child,
    pub args: LlamaServerArgs,
    pub log: Arc<LogTail>,
    pub started_at: std::time::Instant,
}

impl ServerProcess {
    pub fn spawn(args: LlamaServerArgs, log: Arc<LogTail>) -> Result<Self> {
        if !args.binary.is_file() { bail!("server binary missing: {}", args.binary.display()); }
        if !args.model.is_file() { bail!("model file missing: {}", args.model.display()); }
        let mut cmd = Command::new(&args.binary);
        cmd.args(&args.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(dir) = args.binary.parent() { cmd.current_dir(dir); }
        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn().with_context(|| format!("spawning {}", args.binary.display()))?;
        for (name, stream) in [("stdout", child.stdout.take().map(|s| Box::pin(s) as _)), ("stderr", child.stderr.take().map(|s| Box::pin(s) as _))] {
            let stream: Option<std::pin::Pin<Box<dyn tokio::io::AsyncRead + Send>>> = stream;
            if let Some(s) = stream {
                let log = log.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(s).lines();
                    while let Ok(Some(l)) = lines.next_line().await {
                        tracing::trace!(target: "llama-server", stream = name, "{l}");
                        log.push(l);
                    }
                });
            }
        }
        Ok(Self { child, args, log, started_at: std::time::Instant::now() })
    }

    pub fn pid(&self) -> Option<u32> { self.child.id() }

    /// Non-blocking exit check.
    pub fn try_exit_status(&mut self) -> Option<std::process::ExitStatus> { self.child.try_wait().ok().flatten() }

    pub async fn kill(&mut self) {
        let _ = self.child.start_kill();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), self.child.wait()).await;
    }

    /// Detect an out-of-memory failure from the log tail.
    pub fn looks_like_oom(&self) -> bool {
        ["out of memory", "cudaMalloc failed", "CUDA error: out of memory", "failed to allocate", "ggml_backend_cuda_buffer_type_alloc_buffer"]
            .iter().any(|n| self.log.contains(n))
    }
}

pub fn model_display(p: &Path) -> String { p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default() }
