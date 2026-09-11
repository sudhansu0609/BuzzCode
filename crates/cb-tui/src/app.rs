//! TUI application state + event loop.

use crate::blocks::{Block, TranscriptBlock};
use anyhow::Result;
use cb_core::events::AgentEvent;
use cb_core::handle::{AgentCmd, AgentHandle};
use cb_core::permission::{PermissionDecision, PermissionRequest};
use cb_core::session::Session;
use cb_core::task::{normalize_effort, TaskClass};
use cb_core::PermissionMode;
use cb_engine::EngineState;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use crate::input::InputBox;

const PLACEHOLDER: &str = "Ask buzzcode… (Enter to send, Shift+Enter for newline, /help)";

pub struct TuiOptions {
    pub session: Arc<Session>,
    pub handle: AgentHandle,
    pub plans: Arc<cb_core::plan::PlanStore>,
    pub events_rx: mpsc::Receiver<AgentEvent>,
    pub perm_rx: mpsc::Receiver<PermissionRequest>,
}

pub struct PendingPermission {
    pub req: PermissionRequest,
    pub scroll: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay { None, Permission, Help, EngineLog }

pub struct Stats {
    pub ctx_used: u32,
    pub ctx_total: u32,
    pub decode_tps: f64,
    pub cache_ratio: f64,
    pub prompt_ms: f64,
    pub mtp: Option<f64>,
    pub turn: u32,
    pub max_turns: u32,
    pub effort: String,
    pub task: TaskClass,
    /// Live generation counters for the current turn (≈ one streamed chunk per token).
    pub gen_tokens: u32,
    pub gen_start: Option<Instant>,
    pub turn_start: Option<Instant>,
    /// Session totals.
    pub total_gen_tokens: u64,
    pub total_gen_secs: f64,
}

impl Stats {
    /// tok/s of the generation in progress (None before the first token).
    pub fn live_tps(&self) -> Option<f64> {
        let t0 = self.gen_start?;
        let s = t0.elapsed().as_secs_f64();
        if s < 0.25 || self.gen_tokens < 2 { return None; }
        Some(self.gen_tokens as f64 / s)
    }
    pub fn session_avg_tps(&self) -> Option<f64> {
        if self.total_gen_secs > 0.0 && self.total_gen_tokens > 0 { Some(self.total_gen_tokens as f64 / self.total_gen_secs) } else { None }
    }
}

pub struct App {
    pub session: Arc<Session>,
    pub handle: AgentHandle,
    pub plans: Arc<cb_core::plan::PlanStore>,
    pub blocks: Vec<TranscriptBlock>,
    pub input: InputBox,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
    pub scroll_from_bottom: usize,
    pub follow: bool,
    pub show_reasoning: bool,
    pub overlay: Overlay,
    pub pending: Option<PendingPermission>,
    pub perm_queue: std::collections::VecDeque<PermissionRequest>,
    pub engine_state: EngineState,
    pub stats: Stats,
    pub busy: bool,
    pub dirty: bool,
    pub quit: bool,
    pub last_ctrl_c: Option<Instant>,
    pub spinner: usize,
    pub max_blocks: usize,
    pub flash: Option<(String, Instant)>,
    pub factory: crate::factory::Factory,
    pub show_factory: bool,
    pub arcade: Option<crate::arcade::ArcadeServer>,
    pub last_snapshot: String,
    /// Name of the subagent whose events are currently arriving (for transcript labels).
    pub current_worker: Option<String>,
}

impl App {
    fn new(session: Arc<Session>, handle: AgentHandle, plans: Arc<cb_core::plan::PlanStore>) -> Self {
        let cfg = session.cfg.clone();
        let input = InputBox::with_placeholder(PLACEHOLDER);
        let profile = session.engine.profile();
        let stats = Stats { ctx_used: 0, ctx_total: session.engine.n_ctx(), decode_tps: 0.0, cache_ratio: 0.0, prompt_ms: 0.0, mtp: None, turn: 0, max_turns: cfg.general.max_turns, effort: profile.effort.chat.clone(), task: TaskClass::Chat,
            gen_tokens: 0, gen_start: None, turn_start: None, total_gen_tokens: 0, total_gen_secs: 0.0 };
        let engine_state = session.engine.state();
        Self {
            session, handle, plans, blocks: Vec::new(), input, history: Vec::new(), history_pos: None, scroll_from_bottom: 0, follow: true,
            show_reasoning: cfg.tui.show_reasoning, overlay: Overlay::None, pending: None, perm_queue: Default::default(),
            engine_state, stats, busy: false, dirty: true, quit: false, last_ctrl_c: None, spinner: 0, max_blocks: cfg.tui.max_transcript_blocks, flash: None,
            factory: crate::factory::Factory::default(), show_factory: cfg.tui.show_factory, current_worker: None,
            arcade: None, last_snapshot: String::new(),
        }
    }

    /// Push the current factory state to the browser page (cheap; coalesced by content).
    pub fn publish_arcade(&mut self) {
        let Some(a) = &self.arcade else { return };
        let json = crate::arcade::snapshot_json(&self.factory, &self.session.engine.alias(), self.stats.live_tps().unwrap_or(self.stats.decode_tps), self.stats.ctx_used, self.stats.ctx_total, self.busy);
        if json != self.last_snapshot { a.publish(json.clone()); self.last_snapshot = json; }
    }

    pub fn push(&mut self, b: Block) {
        self.blocks.push(TranscriptBlock::new(b));
        if self.blocks.len() > self.max_blocks { self.blocks.drain(0..self.blocks.len() - self.max_blocks); }
        self.dirty = true;
    }

    fn last_assistant_mut(&mut self) -> Option<&mut TranscriptBlock> {
        self.blocks.iter_mut().rev().find(|b| matches!(b.block, Block::Assistant { streaming: true, .. }))
    }

    pub fn apply_event(&mut self, ev: AgentEvent) {
        self.dirty = true;
        self.factory.apply(&ev);
        match ev {
            AgentEvent::AgentSpawned { id, kind, task } => {
                let name = self.factory.workers.iter().find(|w| w.id == id).map(|w| w.name.clone()).unwrap_or_else(|| format!("#{id}"));
                self.current_worker = Some(name.clone());
                self.show_factory = true;
                self.push(Block::Note(format!("▸ {name} joins as a {kind} player: {}", task.lines().next().unwrap_or("").chars().take(160).collect::<String>())));
                return;
            }
            AgentEvent::AgentFinished { id, outcome, turns } => {
                let name = self.factory.workers.iter().find(|w| w.id == id).map(|w| w.name.clone()).unwrap_or_else(|| format!("#{id}"));
                self.current_worker = None;
                self.push(Block::Note(format!("▸ {name} finished ({outcome}, {turns} turns) — report handed to P1")));
                return;
            }
            _ => {}
        }
        match ev {
            AgentEvent::TurnStart { turn, task, effort } => {
                self.stats.turn = turn; self.stats.task = task; self.stats.effort = effort; self.busy = true;
                self.stats.gen_tokens = 0; self.stats.gen_start = None; self.stats.turn_start = Some(Instant::now());
            }
            AgentEvent::ReasoningDelta { text } => {
                self.count_token();
                match self.last_assistant_mut() {
                    Some(b) => { if let Block::Assistant { reasoning, .. } = &mut b.block { reasoning.push_str(&text); } b.invalidate(); }
                    None => self.push(Block::Assistant { content: String::new(), reasoning: text, streaming: true }),
                }
            }
            AgentEvent::ContentDelta { text } => {
                self.count_token();
                match self.last_assistant_mut() {
                    Some(b) => { if let Block::Assistant { content, .. } = &mut b.block { content.push_str(&text); } b.invalidate(); }
                    None => self.push(Block::Assistant { content: text, reasoning: String::new(), streaming: true }),
                }
            }
            AgentEvent::AssistantMessage { content, .. } => {
                match self.last_assistant_mut() {
                    Some(b) => { if let Block::Assistant { content: c, streaming, .. } = &mut b.block { *c = content; *streaming = false; } b.invalidate(); }
                    None => if !content.is_empty() { self.push(Block::Assistant { content, reasoning: String::new(), streaming: false }); },
                }
            }
            AgentEvent::ToolCallStart { name, arguments, .. } => {
                let args = compact_args(&arguments);
                let name = match &self.current_worker { Some(w) => format!("{w} › {name}"), None => name };
                self.push(Block::Tool { name, args, output: String::new(), is_error: false, done: false, collapsed: true, truncated: false });
            }
            AgentEvent::ToolCallResult { name, output, is_error, truncated, .. } => {
                let name = match &self.current_worker { Some(w) => format!("{w} › {name}"), None => name };
                if let Some(b) = self.blocks.iter_mut().rev().find(|b| matches!(&b.block, Block::Tool { name: n, done: false, .. } if *n == name)) {
                    if let Block::Tool { output: o, is_error: e, done, truncated: t, collapsed, .. } = &mut b.block { *o = output; *e = is_error; *done = true; *t = truncated; *collapsed = !is_error; }
                    b.invalidate();
                }
            }
            AgentEvent::PermissionRequested { .. } | AgentEvent::PermissionResolved { .. } => {}
            AgentEvent::Verify { command, ok, output } => self.push(Block::Verify { command, ok, output, collapsed: true }),
            AgentEvent::Compaction { messages_before, messages_after, summary_tokens } => self.push(Block::Note(format!("↻ context compacted: {messages_before} → {messages_after} messages ({summary_tokens}-token summary)"))),
            AgentEvent::Metrics { timings, ctx_used, ctx_total, cache_ratio, decode_tps } => {
                self.stats.ctx_used = ctx_used; self.stats.ctx_total = ctx_total; self.stats.cache_ratio = cache_ratio; self.stats.decode_tps = decode_tps;
                self.stats.prompt_ms = timings.prompt_ms; self.stats.mtp = timings.draft_acceptance();
                // Server-reported numbers are exact: fold them into the session average.
                self.stats.total_gen_tokens += timings.predicted_n as u64;
                self.stats.total_gen_secs += timings.predicted_ms / 1000.0;
                // Next generation (tool loop) starts fresh counters.
                self.stats.gen_tokens = 0; self.stats.gen_start = None;
            }
            AgentEvent::Warning { text } => self.push(Block::Warn(text)),
            AgentEvent::Error { text } => self.push(Block::Error(text)),
            AgentEvent::Finished { reason, turns } => {
                self.busy = false;
                if let Some(b) = self.last_assistant_mut() { if let Block::Assistant { streaming, .. } = &mut b.block { *streaming = false; } b.invalidate(); }
                if reason == "model_switch" { self.stats.ctx_total = self.session.engine.n_ctx(); self.stats.ctx_used = 0; }
                else if reason != "end_turn" { self.push(Block::Note(format!("— stopped: {reason} after {turns} turns"))); }
            }
            AgentEvent::AgentSpawned { .. } | AgentEvent::AgentFinished { .. } => {}
        }
    }

    fn count_token(&mut self) {
        if self.stats.gen_start.is_none() { self.stats.gen_start = Some(Instant::now()); }
        self.stats.gen_tokens += 1;
    }

    fn submit(&mut self) {
        let text = self.input.text();
        // If line ends with trailing backslash, treat as multiline continuation
        if text.trim_end().ends_with('\\') {
            let trimmed = text.trim_end();
            let without_slash = &trimmed[..trimmed.len() - 1];
            self.input.set_text(without_slash);
            self.input.newline();
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() { return; }
        self.input.clear();
        self.history.push(text.clone());
        self.history_pos = None;
        if text.starts_with('/') { self.slash(&text); return; }
        if self.busy { self.flash("agent is busy — Ctrl+C to abort first"); return; }
        self.push(Block::User(text.clone()));
        self.follow = true; self.scroll_from_bottom = 0;
        self.busy = true;
        self.factory.set_big_task(&text);
        if !self.handle.try_send(AgentCmd::UserMessage(text)) { self.push(Block::Error("agent task is gone".into())); self.busy = false; }
    }

    fn flash(&mut self, s: &str) { self.flash = Some((s.to_string(), Instant::now())); self.dirty = true; }

    fn slash(&mut self, cmd: &str) {
        let mut parts = cmd.trim_start_matches('/').splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim().to_string();
        match name {
            "help" | "?" => self.overlay = Overlay::Help,
            "quit" | "exit" | "q" => self.quit = true,
            "clear" => { self.blocks.clear(); self.flash("transcript cleared (context unchanged; use a new session to reset the model context)"); }
            "mode" => {
                let m = match arg.as_str() {
                    "ask" | "manual" => Some(PermissionMode::Ask),
                    "edits" | "auto" => Some(PermissionMode::AutoAcceptEdits),
                    "yolo" | "all" | "allow-all" | "allowall" | "allow" => Some(PermissionMode::Yolo),
                    _ => None,
                };
                match m { Some(m) => self.set_permission_mode(m, false), None => self.flash("usage: /mode ask|edits|yolo (or /allow all)") }
            }
            "allow" | "allow-all" | "allowall" | "yolo" => self.set_permission_mode(PermissionMode::Yolo, arg == "save"),
            "manual" | "ask" => self.set_permission_mode(PermissionMode::Ask, arg == "save"),
            "auto" | "auto-edits" => self.set_permission_mode(PermissionMode::AutoAcceptEdits, arg == "save"),
            "model" | "models" => self.model_command(&arg),
            "ctx" | "context" => self.ctx_command(&arg),
            "high" => {
                let _ = self.handle.try_send(AgentCmd::SetEffort(Some("xhigh".to_string())));
                self.stats.effort = "high".into();
                self.flash("reasoning mode: high (xhigh)");
            }
            "effort" => match normalize_effort(&arg) {
                Some(e) => {
                    let display = if e == "xhigh" { "high" } else { e };
                    let _ = self.handle.try_send(AgentCmd::SetEffort(Some(e.to_string())));
                    self.stats.effort = display.into();
                    self.flash(&format!("reasoning effort: {display}"));
                }
                None => self.flash("usage: /effort high|medium|low|none"),
            },
            "compact" => { let _ = self.handle.try_send(AgentCmd::Compact); self.flash("compacting…"); }
            "reasoning" | "think" => {
                if !arg.is_empty() {
                    match normalize_effort(&arg) {
                        Some(e) => {
                            let display = if e == "xhigh" { "high" } else { e };
                            let _ = self.handle.try_send(AgentCmd::SetEffort(Some(e.to_string())));
                            self.stats.effort = display.into();
                            self.flash(&format!("reasoning mode: {display}"));
                        }
                        None => self.flash("usage: /reasoning high|medium|low|none"),
                    }
                } else {
                    self.show_reasoning = !self.show_reasoning;
                    for b in &mut self.blocks { b.invalidate(); }
                }
            }
            "factory" | "agents" | "workers" => { self.show_factory = !self.show_factory; self.flash(if self.show_factory { "factory floor on (F3 toggles)" } else { "factory floor off" }); }
            "arcade" | "game" => match &self.arcade {
                Some(a) => { a.open_browser(); self.flash(&format!("BUZZCODE ARCADE → {}", a.url())); }
                None => self.flash("arcade server is not running (tui.arcade_port = 0?)"),
            },
            "engine" => match arg.as_str() {
                "log" | "" => self.overlay = Overlay::EngineLog,
                "status" => { let s = self.session.engine.state(); self.push(Block::Note(format!("engine: {s:?} · slot switches {}", self.session.engine.slot().switch_count()))); }
                "restart" => { let e = self.session.engine.clone(); tokio::spawn(async move { let _ = e.restart().await; }); self.flash("restarting engine…"); }
                _ => self.flash("usage: /engine log|status|restart"),
            },
            "cost" | "stats" => {
                let h = &self.handle; let _ = h;
                let s = &self.stats;
                self.push(Block::Note(format!("ctx {}/{} ({:.0}%) · last decode {:.1} tok/s · cache {:.0}% · prompt {:.0} ms · turn {}/{} · effort {} · engine {:?}",
                    s.ctx_used, s.ctx_total, 100.0 * s.ctx_used as f64 / s.ctx_total.max(1) as f64, s.decode_tps, s.cache_ratio * 100.0, s.prompt_ms, s.turn, s.max_turns, s.effort, self.engine_state)));
            }
            "plan" => {
                if self.busy { self.flash("agent is busy"); return; }
                let first = if arg.is_empty() { None } else { Some(arg.clone()) };
                self.push(Block::Note("entering PLAN mode: read-only tools + write_plan; the agent investigates and submits a plan. Then /act to implement or reply with feedback.".into()));
                if first.is_some() { self.push(Block::User(arg.clone())); self.busy = true; }
                let _ = self.handle.try_send(AgentCmd::SwitchMode { mode: cb_core::AgentMode::Plan, first_message: first });
                self.stats.task = TaskClass::Plan;
            }
            "act" => {
                if self.busy { self.flash("agent is busy"); return; }
                let latest = self.plans.latest.lock().clone();
                match latest {
                    Some((plan, path)) => {
                        let extra = if arg.is_empty() { String::new() } else { format!("\n\nAdditional instructions from the user: {arg}") };
                        let msg = format!("Implement the following plan step by step. Verify each step as described. Plan file: {}\n\n{}{extra}", path.display(), plan.to_markdown());
                        self.push(Block::Note(format!("entering ACT mode with plan \"{}\" ({} steps)", plan.title, plan.steps.len())));
                        self.push(Block::User(format!("(implement plan: {})", plan.title)));
                        self.busy = true;
                        let _ = self.handle.try_send(AgentCmd::SwitchMode { mode: cb_core::AgentMode::Main, first_message: Some(msg) });
                        self.stats.task = TaskClass::Chat;
                    }
                    None => {
                        self.push(Block::Note("no plan yet — switching to normal mode with a fresh context".into()));
                        let _ = self.handle.try_send(AgentCmd::SwitchMode { mode: cb_core::AgentMode::Main, first_message: if arg.is_empty() { None } else { Some(arg.clone()) } });
                        if !arg.is_empty() { self.push(Block::User(arg.clone())); self.busy = true; }
                        self.stats.task = TaskClass::Chat;
                    }
                }
            }
            "new" | "reset" => {
                if self.busy { self.flash("agent is busy"); return; }
                self.push(Block::Note("new session: fresh context (same engine)".into()));
                let _ = self.handle.try_send(AgentCmd::SwitchMode { mode: cb_core::AgentMode::Main, first_message: None });
            }
            "map" | "index" => self.flash("use the repo_map / symbol_search tools via the agent, or `buzzcode index map` in a shell"),
            _ => self.flash(&format!("unknown command /{name}; try /help")),
        }
    }

    fn set_permission_mode(&mut self, m: PermissionMode, save: bool) {
        self.session.perms.set_mode(m);
        let label = match m { PermissionMode::Yolo => "ALLOW-ALL: every tool runs without asking", PermissionMode::Ask => "MANUAL: edits and commands ask for permission", PermissionMode::AutoAcceptEdits => "AUTO-EDITS: file edits run, shell commands ask" };
        if save {
            let name = match m { PermissionMode::Yolo => "yolo", PermissionMode::Ask => "ask", PermissionMode::AutoAcceptEdits => "auto_accept_edits" };
            match cb_config::persist_permission_mode(&self.session.cfg, name) { Ok(p) => self.push(Block::Note(format!("{label} (saved as default in {})", p.display()))), Err(e) => self.push(Block::Error(format!("could not save: {e:#}"))) }
        } else {
            self.push(Block::Note(format!("{label}  (add `save` to make it the default)")));
        }
    }

    /// `/ctx` → show; `/ctx 64k [save]` → restart the engine with a different preferred context.
    fn ctx_command(&mut self, arg: &str) {
        let engine = self.session.engine.clone();
        let plan = engine.plan();
        if arg.is_empty() {
            let p = engine.profile();
            self.push(Block::Note(format!("context: running {} tokens (profile prefers {}, min {}){}\nKV cache ≈ {} per token → {:.2} GiB at this size. Use /ctx 48k, /ctx 64k … (add `save` to persist). Larger context = less VRAM headroom; the planner shrinks it if it doesn't fit.",
                engine.n_ctx(), p.ctx, p.ctx_min, plan.as_ref().map(|pl| format!(" · plan est {:.2} GiB VRAM", pl.est_vram_gib())).unwrap_or_default(),
                engine.facts().map(|f| { let kv = 2 * f.n_attn_layers as u64 * f.n_head_kv as u64 * (f.head_dim + f.value_dim) as u64 * 17 / 16; format!("{kv} B") }).unwrap_or("?".into()),
                engine.facts().map(|f| (2 * f.n_attn_layers as u64 * f.n_head_kv as u64 * (f.head_dim + f.value_dim) as u64 * 17 / 16) as f64 * engine.n_ctx() as f64 / (1u64 << 30) as f64).unwrap_or(0.0))));
            return;
        }
        if self.busy { self.flash("agent is busy — finish or abort first"); return; }
        let mut parts = arg.split_whitespace();
        let val = parts.next().unwrap_or("");
        let save = parts.next() == Some("save");
        let t = val.to_ascii_lowercase();
        let n: Option<u64> = if let Some(k) = t.strip_suffix('k') { k.parse::<f64>().ok().map(|v| (v * 1024.0) as u64) } else { t.parse().ok() };
        let Some(n) = n.filter(|n| (2048..=1_048_576).contains(n)) else { self.flash("usage: /ctx 48k | /ctx 65536 [save]"); return; };
        let n = ((n + 255) / 256 * 256) as u32;
        let mut profile = engine.profile();
        profile.ctx = n; profile.ctx_min = profile.ctx_min.min(n);
        if save { match cb_config::persist_profile(&self.session.cfg, &profile, true) { Ok(p) => self.push(Block::Note(format!("saved ctx {n} in {}", p.display()))), Err(e) => self.push(Block::Warn(format!("could not save: {e:#}"))) } }
        self.push(Block::Note(format!("restarting engine with preferred context {n} (fresh conversation context)…")));
        let tx = self.handle.sender();
        let session = self.session.clone();
        self.busy = true;
        tokio::spawn(async move {
            match engine.switch_profile(profile).await {
                Ok(()) => {
                    session.emit(AgentEvent::Warning { text: format!("engine ready: ctx {} (planner{})", engine.n_ctx(), if engine.n_ctx() < n { " reduced it to fit VRAM" } else { "" }) });
                    let _ = tx.send(AgentCmd::SwitchMode { mode: cb_core::AgentMode::Main, first_message: None }).await;
                }
                Err(e) => session.emit(AgentEvent::Error { text: format!("engine restart failed: {e:#}") }),
            }
            session.emit(AgentEvent::Finished { reason: "model_switch".into(), turns: 0 });
        });
    }

    /// `/model` → ranked table; `/model <#|name|path>` → switch engine + fresh context (and save as default).
    fn model_command(&mut self, arg: &str) {
        if self.busy { self.flash("agent is busy — finish or abort first"); return; }
        let cfg = self.session.cfg.clone();
        let current = self.session.engine.profile().name;
        let (gpu, cands) = match cb_engine::catalog::recommend(&cfg) { Ok(v) => v, Err(e) => { self.push(Block::Error(format!("model scan failed: {e:#}"))); return; } };
        if arg.is_empty() || arg == "list" {
            self.push(Block::Note(cb_engine::catalog::render_table(&gpu, &cands, Some(&current)) + "\nswitch with /model <#|name>  ·  /model auto picks the ★ recommendation"));
            return;
        }
        let sel = if arg == "auto" || arg == "best" { cands.first() } else { cb_engine::catalog::select(&cands, &cfg, arg) };
        let Some(c) = sel else { self.push(Block::Error(format!("no model matches {arg:?}; /model to list"))); return; };
        let profile = cb_engine::catalog::profile_for(&cfg, c);
        if profile.name == current { self.flash("that model is already active"); return; }
        match cb_config::persist_profile(&cfg, &profile, true) {
            Ok(p) => self.push(Block::Note(format!("switching to {} — {} (~{:.0} tok/s){}; saved as default in {}", c.name, c.note, c.est_tps, if c.is_local() { "" } else { ", downloading first" }, p.display()))),
            Err(e) => self.push(Block::Warn(format!("could not save default: {e:#}"))),
        }
        self.stats.ctx_total = profile.ctx;
        let engine = self.session.engine.clone();
        let tx = self.handle.sender();
        let session = self.session.clone();
        self.busy = true;
        tokio::spawn(async move {
            match engine.switch_profile(profile).await {
                Ok(()) => {
                    session.emit(AgentEvent::Warning { text: format!("engine ready: {} · ctx {}", engine.alias(), engine.n_ctx()) });
                    let _ = tx.send(AgentCmd::SwitchMode { mode: cb_core::AgentMode::Main, first_message: None }).await;
                }
                Err(e) => session.emit(AgentEvent::Error { text: format!("model switch failed: {e:#}") }),
            }
            session.emit(AgentEvent::Finished { reason: "model_switch".into(), turns: 0 });
        });
    }

    fn handle_key(&mut self, key: KeyEvent) {
        self.dirty = true;
        // Overlays first.
        match self.overlay {
            Overlay::Permission => { self.permission_key(key); return; }
            Overlay::Help | Overlay::EngineLog => { if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q')) || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) { self.overlay = Overlay::None; } return; }
            Overlay::None => {}
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy { self.handle.abort(); self.flash("aborting…"); self.last_ctrl_c = Some(Instant::now()); return; }
                match self.last_ctrl_c { Some(t) if t.elapsed() < Duration::from_secs(2) => self.quit = true, _ => { self.last_ctrl_c = Some(Instant::now()); self.flash("press Ctrl+C again to quit"); } }
            }
            KeyCode::Char('d') if ctrl => self.quit = true,
            KeyCode::Char('t') if ctrl => { self.show_reasoning = !self.show_reasoning; for b in &mut self.blocks { b.invalidate(); } }
            KeyCode::Char('o') if ctrl => { if let Some(b) = self.blocks.iter_mut().rev().find(|b| matches!(b.block, Block::Tool { .. } | Block::Verify { .. })) { match &mut b.block { Block::Tool { collapsed, .. } | Block::Verify { collapsed, .. } => *collapsed = !*collapsed, _ => {} } b.invalidate(); } }
            KeyCode::F(1) => self.overlay = Overlay::Help,
            KeyCode::F(2) => self.overlay = Overlay::EngineLog,
            KeyCode::F(3) => self.show_factory = !self.show_factory,
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) || key.modifiers.contains(KeyModifiers::ALT) || key.modifiers.contains(KeyModifiers::CONTROL) => { self.input.newline(); }
            KeyCode::Char('\n') | KeyCode::Char('j') if ctrl => { self.input.newline(); }
            KeyCode::Enter => self.submit(),
            KeyCode::PageUp => { self.follow = false; self.scroll_from_bottom += 10; }
            KeyCode::PageDown => { self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(10); if self.scroll_from_bottom == 0 { self.follow = true; } }
            KeyCode::End if ctrl => { self.scroll_from_bottom = 0; self.follow = true; }
            KeyCode::Up if (self.input.is_empty() && !self.history.is_empty()) || self.history_pos.is_some() => {
                if self.input.handle_key(key) { return; } // moved within multi-line text
                let pos = match self.history_pos { None => self.history.len() - 1, Some(p) => p.saturating_sub(1) };
                self.history_pos = Some(pos);
                self.input.set_text(&self.history[pos].clone());
            }
            KeyCode::Down if self.history_pos.is_some() => {
                if self.input.handle_key(key) { return; }
                let pos = self.history_pos.unwrap() + 1;
                if pos >= self.history.len() { self.history_pos = None; self.input.clear(); } else { self.history_pos = Some(pos); self.input.set_text(&self.history[pos].clone()); }
            }
            _ => { self.input.handle_key(key); }
        }
    }

    fn permission_key(&mut self, key: KeyEvent) {
        let Some(p) = self.pending.as_mut() else { self.overlay = Overlay::None; return; };
        let decision = match key.code {
            KeyCode::Char('y') | KeyCode::Enter => Some(PermissionDecision::Allow),
            KeyCode::Char('a') => Some(PermissionDecision::AllowAlways),
            KeyCode::Char('A') => {
                self.set_permission_mode(PermissionMode::Yolo, false);
                Some(PermissionDecision::Allow)
            }
            KeyCode::Char('n') | KeyCode::Esc => Some(PermissionDecision::Deny),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => { self.handle.abort(); Some(PermissionDecision::Deny) }
            KeyCode::Down | KeyCode::Char('j') => { p.scroll += 1; None }
            KeyCode::Up | KeyCode::Char('k') => { p.scroll = p.scroll.saturating_sub(1); None }
            KeyCode::PageDown => { p.scroll += 20; None }
            KeyCode::PageUp => { p.scroll = p.scroll.saturating_sub(20); None }
            _ => None,
        };
        if let Some(d) = decision {
            let p = self.pending.take().unwrap();
            let _ = p.req.respond.send(d);
            self.next_permission();
        }
    }

    fn next_permission(&mut self) {
        if let Some(req) = self.perm_queue.pop_front() {
            self.pending = Some(PendingPermission { req, scroll: 0 });
            self.overlay = Overlay::Permission;
        } else {
            self.overlay = Overlay::None;
        }
        self.dirty = true;
    }

    pub fn enqueue_permission(&mut self, req: PermissionRequest) {
        self.perm_queue.push_back(req);
        if self.pending.is_none() { self.next_permission(); }
    }
}

fn compact_args(v: &serde_json::Value) -> String {
    match v.as_object() {
        Some(o) => o.iter().map(|(k, v)| {
            let s = match v { serde_json::Value::String(s) => s.clone(), other => other.to_string() };
            let s: String = s.replace('\n', "⏎").chars().take(60).collect();
            format!("{k}={s}")
        }).collect::<Vec<_>>().join(" "),
        None => v.to_string(),
    }
}

pub async fn run(opts: TuiOptions) -> Result<()> {
    let TuiOptions { session, handle, plans, mut events_rx, mut perm_rx } = opts;
    let mut engine_rx = session.engine.watch_state();
    let mut app = App::new(session.clone(), handle, plans);
    app.push(Block::Note(format!("buzzcode · {} · ctx {} · mode {:?} · /help for keys", session.engine.profile().alias, session.engine.n_ctx(), session.perms.mode())));
    if session.cfg.tui.arcade_port > 0 {
        match crate::arcade::ArcadeServer::start(session.cfg.tui.arcade_port).await {
            Ok(a) => {
                app.push(Block::Note(format!("🕹 BUZZCODE ARCADE live at {}  (/arcade opens it)", a.url())));
                if session.cfg.tui.arcade_auto_open { a.open_browser(); }
                app.arcade = Some(a);
                app.publish_arcade();
            }
            Err(e) => app.push(Block::Warn(format!("arcade server failed: {e:#}"))),
        }
    }

    let mut terminal = ratatui::init();
    let mut events = EventStream::new();
    let tick = Duration::from_millis(session.cfg.tui.render_interval_ms.max(16));
    let mut ticker = tokio::time::interval(tick);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut last_publish = Instant::now();
    let result = loop {
        if app.dirty && last_publish.elapsed() >= Duration::from_millis(100) { app.publish_arcade(); last_publish = Instant::now(); }
        if app.dirty {
            let draw = terminal.draw(|f| crate::ui::draw(f, &mut app));
            if let Err(e) = draw { break Err(e.into()); }
            app.dirty = false;
        }
        if app.quit { break Ok(()); }
        tokio::select! {
            _ = ticker.tick() => {
                if app.busy { app.spinner = app.spinner.wrapping_add(1); app.dirty = true; }
                else if app.show_factory && app.spinner % 15 == 0 { app.spinner = app.spinner.wrapping_add(1); app.dirty = true; } else if app.show_factory { app.spinner = app.spinner.wrapping_add(1); }
                if let Some((_, t)) = &app.flash { if t.elapsed() > Duration::from_secs(4) { app.flash = None; app.dirty = true; } }
            }
            ev = events.next() => {
                match ev {
                    Some(Ok(Event::Key(k))) if k.kind != crossterm::event::KeyEventKind::Release => app.handle_key(k),
                    Some(Ok(Event::Resize(_, _))) => app.dirty = true,
                    Some(Ok(Event::Paste(s))) => { app.input.insert_str(&s); app.dirty = true; }
                    Some(Err(e)) => break Err(e.into()),
                    None => break Ok(()),
                    _ => {}
                }
            }
            ev = events_rx.recv() => {
                match ev { Some(ev) => app.apply_event(ev), None => break Ok(()) }
                // Drain a burst of deltas before redrawing.
                while let Ok(ev) = events_rx.try_recv() { app.apply_event(ev); }
            }
            req = perm_rx.recv() => { if let Some(req) = req { app.enqueue_permission(req); } }
            _ = engine_rx.changed() => { app.engine_state = engine_rx.borrow().clone(); app.dirty = true; }
        }
    };
    ratatui::restore();
    let _ = app.handle.send(AgentCmd::Shutdown).await;
    result
}
