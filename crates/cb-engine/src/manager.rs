//! EngineManager: the one object the rest of the harness talks to.
//!
//! start() = resolve binary + model → inspect GGUF → plan VRAM → spawn → wait /health →
//!           (OOM? shrink + retry) → warm up → persist tuned plan.

use crate::client::{ChatRequest, LlamaClient};
use crate::gguf::{GgufFile, ModelFacts};
use crate::server::{LlamaServerArgs, LogTail, ServerProcess};
use crate::slot::EngineSlot;
use crate::tune::{self, TunedProfile};
use crate::vram::{KvType, PlanOpts, PlanPrefs, VramPlan, VramPlanner};
use crate::{discover, nvidia, ports};
use anyhow::{bail, Context, Result};
use cb_config::{Config, Profile};
use parking_lot::Mutex;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;

#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    Stopped,
    Resolving,
    Starting { attempt: u32 },
    Loading { elapsed_s: u64 },
    Ready { n_ctx: u32, model: String },
    Crashed { attempts: u32, reason: String },
    Restarting,
    External,
}

impl EngineState {
    pub fn is_ready(&self) -> bool { matches!(self, EngineState::Ready { .. } | EngineState::External) }
}

pub struct EngineManager {
    cfg: Arc<Config>,
    profile: parking_lot::RwLock<Profile>,
    /// The port the server is actually on. `engine.port` in config.toml is only
    /// the preferred one (GUARDIAN_PLAN.md section 11): if it is busy at start()
    /// we step forward, and from then on this — not the config — is the truth.
    port: Mutex<u16>,
    /// Rebuilt whenever the port moves, so every caller of `client()` talks to
    /// the server we actually spawned.
    client: parking_lot::RwLock<LlamaClient>,
    /// True while our entry is in the shared ledger, so `stop()` withdraws
    /// exactly what `start()` published and nothing else.
    published: Mutex<bool>,
    slot: Arc<EngineSlot>,
    log: Arc<LogTail>,
    state_tx: watch::Sender<EngineState>,
    state_rx: watch::Receiver<EngineState>,
    proc: Mutex<Option<ServerProcess>>,
    facts: Mutex<Option<Arc<ModelFacts>>>,
    plan: Mutex<Option<VramPlan>>,
    model_path: Mutex<Option<PathBuf>>,
    binary: Mutex<Option<PathBuf>>,
    restart_attempts: Mutex<u32>,
    /// Whether start() was ever successful (used by the crash watcher).
    started_once: Mutex<bool>,
}

impl EngineManager {
    pub fn new(cfg: Arc<Config>, profile: Profile) -> Arc<Self> {
        let client = LlamaClient::with_api_key(&profile.base_url(&cfg.engine), profile.resolved_api_key(&cfg.engine));
        let (state_tx, state_rx) = watch::channel(EngineState::Stopped);
        Arc::new(Self {
            port: Mutex::new(cfg.engine.port), published: Mutex::new(false),
            cfg, profile: parking_lot::RwLock::new(profile), client: parking_lot::RwLock::new(client), slot: Arc::new(EngineSlot::new()), log: Arc::new(LogTail::new(400)),
            state_tx, state_rx, proc: Mutex::new(None), facts: Mutex::new(None), plan: Mutex::new(None),
            model_path: Mutex::new(None), binary: Mutex::new(None), restart_attempts: Mutex::new(0), started_once: Mutex::new(false),
        })
    }

    pub fn config(&self) -> &Config { &self.cfg }
    /// Current profile (cloned; it can be swapped at runtime with `switch_profile`).
    pub fn profile(&self) -> Profile { self.profile.read().clone() }
    /// A handle to the server on the port we are actually using. Cloned rather
    /// than borrowed because the port — and with it the base URL — can change
    /// between `start()` calls.
    pub fn client(&self) -> LlamaClient { self.client.read().clone() }
    /// The port the engine is really on (the configured one until `start()`
    /// has had to step forward).
    pub fn port(&self) -> u16 { *self.port.lock() }
    pub fn base_url(&self) -> String { self.client.read().base_url().to_string() }

    /// Point the client at a new port, and remember it.
    fn set_port(&self, port: u16) {
        *self.port.lock() = port;
        *self.client.write() = LlamaClient::with_api_key(&format!("http://{}:{}", self.cfg.engine.host, port), self.cfg.engine.resolved_api_key());
    }
    pub fn slot(&self) -> Arc<EngineSlot> { self.slot.clone() }
    pub fn log(&self) -> Arc<LogTail> { self.log.clone() }
    pub fn state(&self) -> EngineState { self.state_rx.borrow().clone() }
    pub fn watch_state(&self) -> watch::Receiver<EngineState> { self.state_rx.clone() }
    pub fn facts(&self) -> Option<Arc<ModelFacts>> { self.facts.lock().clone() }
    pub fn plan(&self) -> Option<VramPlan> { self.plan.lock().clone() }
    pub fn model_path(&self) -> Option<PathBuf> { self.model_path.lock().clone() }
    /// PID of the `llama-server` we spawned, once there is one. `engine.json`
    /// carries it so `engine stop` can aim at the tree from another process.
    pub fn server_pid(&self) -> Option<u32> { self.proc.lock().as_ref().and_then(|p| p.pid()) }
    pub fn alias(&self) -> String { self.profile.read().request_model() }
    pub fn n_ctx(&self) -> u32 {
        match self.state() {
            EngineState::Ready { n_ctx, .. } => n_ctx,
            EngineState::External => self.profile.read().ctx,
            _ => self.plan().map(|p| p.n_ctx).unwrap_or(self.profile.read().ctx),
        }
    }

    /// Stop the server, switch to another profile/model, and start again.
    pub async fn switch_profile(self: &Arc<Self>, profile: Profile) -> Result<()> {
        self.set_state(EngineState::Restarting);
        self.stop().await;
        *self.profile.write() = profile;
        *self.model_path.lock() = None;
        *self.facts.lock() = None;
        *self.plan.lock() = None;
        *self.restart_attempts.lock() = 0;
        self.start().await
    }
    fn set_state(&self, s: EngineState) { let _ = self.state_tx.send(s); }

    // ------------------------------------------------------------------ resolve

    pub async fn ensure_binary(&self) -> Result<PathBuf> {
        if let Some(b) = self.binary.lock().clone() { return Ok(b); }
        let b = discover::find_server(&self.cfg)?;
        *self.binary.lock() = Some(b.clone());
        Ok(b)
    }

    /// Resolve the profile's model to a local path, downloading from HF if it is `repo:file` and absent.
    pub async fn ensure_model(&self, mut on_progress: Option<crate::download::ProgressFn>) -> Result<PathBuf> {
        if let Some(p) = self.model_path.lock().clone() { return Ok(p); }
        let spec = &self.profile.read().model.clone();
        let path = match discover::find_model(&self.cfg, spec) {
            Some(p) => p,
            None => {
                let (repo, file) = discover::split_repo_file(spec);
                let Some(repo) = repo else {
                    bail!("model {spec:?} not found locally (searched models_dir + extra_search_dirs) and is not a hf `repo:file` spec");
                };
                let dest = self.cfg.engine.models_dir_path();
                let shards = crate::download::shard_names(file);
                let mut first = None;
                for (i, s) in shards.iter().enumerate() {
                    tracing::info!(repo, file = %s, "downloading model shard {}/{}", i + 1, shards.len());
                    let p = crate::download::hf_download(repo, s, &dest, on_progress.take()).await?;
                    if first.is_none() { first = Some(p); }
                }
                first.unwrap()
            }
        };
        let path = discover::first_shard(&path);
        *self.model_path.lock() = Some(path.clone());
        Ok(path)
    }

    pub fn inspect(&self, path: &std::path::Path) -> Result<Arc<ModelFacts>> {
        if let Some(f) = self.facts.lock().clone() { if f.path == path { return Ok(f); } }
        let g = GgufFile::open(path)?;
        let facts = Arc::new(g.facts());
        *self.facts.lock() = Some(facts.clone());
        Ok(facts)
    }

    // ------------------------------------------------------------------ planning

    pub fn plan_opts_for(cfg: &Config, profile: &Profile, has_mtp: bool) -> PlanOpts {
        PlanOpts {
            kv_type: if profile.flash_attn { KvType::parse(&profile.kv_type) } else { KvType::F16 },
            ubatch: profile.ubatch,
            ctx_checkpoints: profile.ctx_checkpoints,
            mtp: profile.spec == "draft-mtp" && has_mtp,
            cuda_context_bytes: cfg.engine.cuda_context_mib << 20,
            safety_bytes: cfg.engine.vram_safety_mib << 20,
        }
    }

    pub fn plan_for(cfg: &Config, profile: &Profile, facts: &ModelFacts, free_vram_bytes: u64, total_vram_bytes: u64) -> VramPlan {
        let (_, free_ram) = nvidia::system_ram();
        let planner = VramPlanner { facts, free_vram: free_vram_bytes, total_vram: total_vram_bytes, free_ram };
        let prefs = PlanPrefs {
            pref_ctx: profile.ctx,
            min_ctx: profile.ctx_min,
            allow_offload: true,
            fixed_ngl: if profile.ngl_auto() { None } else { profile.ngl_value() },
            fixed_ot: if profile.override_tensor.eq_ignore_ascii_case("auto") { None } else { Some(profile.override_tensor.clone()) },
        };
        planner.plan(&prefs, &Self::plan_opts_for(cfg, profile, facts.has_mtp))
    }

    pub fn compute_plan(&self, facts: &ModelFacts, free_vram_bytes: u64, total_vram_bytes: u64) -> VramPlan {
        Self::plan_for(&self.cfg, &self.profile(), facts, free_vram_bytes, total_vram_bytes)
    }

    pub fn is_external(&self) -> bool {
        self.profile.read().is_external(&self.cfg.engine)
    }

    // ------------------------------------------------------------------ lifecycle

    /// Full bring-up. Idempotent: returns quickly if already Ready.
    pub async fn start(self: &Arc<Self>) -> Result<()> {
        if self.state().is_ready() && self.client().health().await.unwrap_or(false) { return Ok(()); }

        if self.is_external() {
            self.set_state(EngineState::Resolving);
            let client = self.client();
            if !client.health().await? { bail!("external engine at {} is not healthy", client.base_url()); }
            let model_name = self.profile.read().request_model();
            let n_ctx = self.profile.read().ctx;
            self.set_state(EngineState::Ready { n_ctx, model: model_name });
            return Ok(());
        }

        self.set_state(EngineState::Resolving);

        // Where this server will actually listen. `engine.port` is a wish
        // (GUARDIAN_PLAN.md section 11 rule 1): if something else holds it —
        // another buzzcode, a stale llama-server, anything — we take the next
        // free port rather than evicting whoever is there. Picked on every
        // start(), because a restart may find the world rearranged.
        let port = ports::pick_port(self.cfg.engine.port, ports::SPAN, &self.cfg.engine.host);
        if port != self.cfg.engine.port {
            tracing::info!(preferred = self.cfg.engine.port, taken = port, "configured port is busy; stepping forward");
        }
        self.set_port(port);

        let binary = self.ensure_binary().await?;
        let model = self.ensure_model(None).await?;
        let facts = self.inspect(&model)?;
        tracing::info!("{}", facts.summary());

        let gpu = nvidia::query().context("querying GPU")?;
        let hash = tune::model_hash(&model)?;
        let mut profile = self.profile();
        // A model without an MTP head can't use draft-mtp: fall back silently.
        if profile.spec == "draft-mtp" && !facts.has_mtp {
            tracing::info!("model has no MTP head; using spec = none");
            profile.spec = "none".into();
            self.profile.write().spec = "none".into();
        }
        let fresh = self.compute_plan(&facts, gpu.mem.free_bytes(), gpu.mem.total_bytes());
        tracing::info!(ngl = fresh.ngl, ctx = fresh.n_ctx, est_gib = fresh.est_vram_gib(), "planned");
        // Reuse a proven tuned plan only if it is at least as good as the fresh plan for the
        // current profile (so raising `ctx` in the profile or freeing VRAM takes effect).
        let mut plan = match tune::load(&self.cfg, &hash, &profile.name) {
            Some(t) if t.successes > 0 && t.still_valid(gpu.mem.free_mib) && profile.ngl_auto()
                && t.n_ctx >= fresh.n_ctx && t.ngl >= fresh.ngl && t.ffn_cpu_blocks <= fresh.ffn_cpu_blocks => {
                tracing::info!("using tuned plan (ngl {}, ctx {})", t.ngl, t.n_ctx);
                t.to_plan()
            }
            _ => fresh,
        };

        // Ask Sentinel before taking the card. This covers every path that
        // spawns a server — `engine serve`, the implicit start behind `-p`, and
        // the TUI — because they all come through here. An absent guardian, or
        // `--force`, returns a grant that books nothing.
        let mut grant = {
            let vram_mib = plan.est_vram_bytes >> 20;
            let ram_mib = self.cfg.engine.cache_ram_mb as u64 + (plan.est_cpu_weight_bytes >> 20);
            crate::sentinel::gate_engine_start(vram_mib, ram_mib).map_err(anyhow::Error::new)?
        };

        let max_attempts = 4;
        for attempt in 1..=max_attempts {
            self.set_state(EngineState::Starting { attempt });
            *self.plan.lock() = Some(plan.clone());
            let args = LlamaServerArgs::build(&self.cfg, &profile, binary.clone(), model.clone(), &plan, port);
            tracing::info!(attempt, "spawning llama-server: {}", args.command_line());
            let proc = ServerProcess::spawn(args, self.log.clone())?;
            *self.proc.lock() = Some(proc);
            match self.wait_healthy(Duration::from_secs(self.cfg.engine.startup_timeout_s)).await {
                Ok(()) => {
                    // `/health` answered: the VRAM is really ours and visible to
                    // Sentinel's own accounting, so the booking can go back.
                    grant.release();
                    let props = self.client().props().await.unwrap_or_default();
                    let n_ctx = if props.n_ctx() > 0 { props.n_ctx() } else { plan.n_ctx };
                    let model_name = crate::server::model_display(&model);
                    self.set_state(EngineState::Ready { n_ctx, model: model_name.clone() });
                    self.slot.reset_owner();
                    // `/health` has answered, so the entry is a promise we can
                    // keep. Now the rest of the family can find the engine
                    // without being told which port it ended up on.
                    self.publish_port(&model_name);
                    *self.started_once.lock() = true;
                    *self.restart_attempts.lock() = 0;
                    // Persist tuned plan with success count.
                    let mut t = tune::load(&self.cfg, &hash, &profile.name).unwrap_or_else(|| TunedProfile::from_plan(&model, &hash, &profile.name, &plan));
                    if t.ngl != plan.ngl || t.n_ctx != plan.n_ctx || t.override_tensor != plan.override_tensor {
                        t = TunedProfile::from_plan(&model, &hash, &profile.name, &plan);
                    }
                    t.successes += 1;
                    if let Ok(g) = nvidia::query() { t.measured_vram_mib = Some(g.mem.used_mib.saturating_sub(gpu.mem.used_mib)); }
                    let _ = tune::save(&self.cfg, &t);
                    self.spawn_crash_watcher();
                    return Ok(());
                }
                Err(e) => {
                    let oom = self.proc.lock().as_ref().map(|p| p.looks_like_oom()).unwrap_or(false);
                    self.stop().await;
                    if oom && attempt < max_attempts {
                        tracing::warn!("start attempt {attempt} hit OOM; shrinking plan");
                        plan = shrink_plan(&facts, plan);
                        continue;
                    }
                    let tail = self.log.snapshot().into_iter().rev().take(15).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
                    self.set_state(EngineState::Crashed { attempts: attempt, reason: e.to_string() });
                    bail!("llama-server failed to become healthy: {e}\n--- last log lines ---\n{tail}");
                }
            }
        }
        bail!("exhausted start attempts")
    }

    async fn wait_healthy(&self, timeout: Duration) -> Result<()> {
        let t0 = Instant::now();
        loop {
            if let Some(p) = self.proc.lock().as_mut() {
                if let Some(st) = p.try_exit_status() { bail!("llama-server exited early: {st}"); }
            }
            if self.client().health().await.unwrap_or(false) { return Ok(()); }
            if t0.elapsed() > timeout { bail!("timeout after {}s waiting for /health", timeout.as_secs()); }
            self.set_state(EngineState::Loading { elapsed_s: t0.elapsed().as_secs() });
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    pub async fn stop(&self) {
        // The ledger entry goes first: for the moment between the kill and the
        // state change, an entry that still points at a dying server is worse
        // than no entry at all.
        self.withdraw_port();
        let proc = self.proc.lock().take();
        if let Some(mut p) = proc { p.kill().await; }
        self.set_state(EngineState::Stopped);
        self.slot.reset_owner();
    }

    /// Announce the engine in the shared ledger. Never fatal: a ledger we
    /// cannot write is a discovery problem, not a startup failure.
    fn publish_port(&self, model: &str) {
        let port = self.port();
        ports::publish(
            ports::APP,
            port,
            &ports::health_url(&self.cfg.engine.host, port),
            ports::engine_extra(&self.alias(), model),
        );
        *self.published.lock() = true;
        tracing::info!(port, "published {} in the port ledger", ports::APP);
    }

    /// Take the entry back out, but only if it was ours to begin with.
    fn withdraw_port(&self) {
        let was = { let mut p = self.published.lock(); std::mem::replace(&mut *p, false) };
        if was { ports::withdraw(ports::APP); }
    }

    pub async fn restart(self: &Arc<Self>) -> Result<()> {
        self.set_state(EngineState::Restarting);
        self.stop().await;
        self.start().await
    }

    fn spawn_crash_watcher(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let exited = { me.proc.lock().as_mut().and_then(|p| p.try_exit_status()) };
                let Some(status) = exited else {
                    if matches!(me.state(), EngineState::Stopped | EngineState::Restarting) && me.proc.lock().is_none() { return; }
                    continue;
                };
                if matches!(me.state(), EngineState::Stopped | EngineState::Restarting) { return; }
                let attempts = { let mut a = me.restart_attempts.lock(); *a += 1; *a };
                tracing::error!(%status, attempts, "llama-server died");
                me.set_state(EngineState::Crashed { attempts, reason: status.to_string() });
                if attempts > me.cfg.engine.restart_max { return; }
                let backoff = Duration::from_secs(1 << (attempts.min(4) - 1));
                tokio::time::sleep(backoff).await;
                me.set_state(EngineState::Restarting);
                let _ = { let p = me.proc.lock().take(); p };
                if let Err(e) = me.start().await { tracing::error!("restart failed: {e:#}"); }
                return; // start() spawns a new watcher on success
            }
        });
    }

    // ------------------------------------------------------------------ warm-up

    /// Prime the KV cache with the session's frozen prefix so the first real turn is a cache hit,
    /// and snapshot it to disk (`warm-<hash>.bin`) for future restarts.
    pub async fn warmup(&self, system_prefix_messages: Vec<serde_json::Value>, tools: Option<serde_json::Value>, effort: &str) -> Result<crate::client::Timings> {
        let mut req = ChatRequest::new(&self.alias(), system_prefix_messages);
        req.tools = tools;
        req.max_tokens = Some(1);
        // Same effort as the session so the rendered system prefix is byte-identical.
        req = req.reasoning_effort(effort);
        if let Some(Value::Object(m)) = &mut req.chat_template_kwargs { m.insert("enable_thinking".into(), Value::Bool(false)); }
        let _g = self.slot.acquire(0).await;
        let (_c, _r, t, _) = self.client().chat_once(req).await?;
        Ok(t)
    }

    pub async fn save_slot(&self, name: &str) -> Result<()> { self.client().slot_save(0, name).await.map(|_| ()) }
    pub async fn restore_slot(&self, name: &str) -> Result<()> { self.client().slot_restore(0, name).await.map(|_| ()) }
}

/// After an OOM: prefer shaving context first, then push FFN blocks to CPU, then reduce ngl.
fn shrink_plan(facts: &ModelFacts, mut plan: VramPlan) -> VramPlan {
    if plan.n_ctx > 16384 {
        plan.n_ctx = (plan.n_ctx - 4096).max(16384);
        plan.rationale.push_str(&format!("\nOOM → ctx {}", plan.n_ctx));
    } else if plan.ffn_cpu_blocks < facts.n_layer * 4 / 10 {
        plan.ffn_cpu_blocks += 4;
        plan.override_tensor = Some(crate::vram::ffn_override_regex(facts.n_layer, plan.ffn_cpu_blocks));
        plan.rationale.push_str(&format!("\nOOM → ffn_cpu_blocks {}", plan.ffn_cpu_blocks));
    } else {
        plan.ngl = plan.ngl.saturating_sub(4);
        plan.override_tensor = None;
        plan.ffn_cpu_blocks = 0;
        plan.rationale.push_str(&format!("\nOOM → ngl {}", plan.ngl));
    }
    plan
}
