//! AgentHandle: run an Agent on its own task and drive it from a UI via channels.

use crate::agent::{Agent, AgentOutcome};
use crate::events::AgentEvent;
use crate::task::TaskClass;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode { Main, Plan }

/// Builds a fresh agent for a mode (new frozen prefix + empty context).
pub type AgentFactory = Arc<dyn Fn(AgentMode) -> futures::future::BoxFuture<'static, anyhow::Result<Agent>> + Send + Sync>;

pub enum AgentCmd {
    UserMessage(String),
    /// Replace the agent with a fresh one in `mode`; optionally send `first_message` immediately.
    SwitchMode { mode: AgentMode, first_message: Option<String> },
    SetEffort(Option<String>),
    SetTask(TaskClass),
    Compact,
    /// Ask for a snapshot of context usage: (used_estimate, n_ctx, compactions, prefix_hash)
    Stats(oneshot::Sender<AgentStats>),
    Shutdown,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStats {
    pub used_tokens: u32,
    pub n_ctx: u32,
    pub compactions: u32,
    pub messages: usize,
    pub prefix_hash: String,
    pub turn: u32,
    pub slot_switches: u64,
}

pub struct AgentHandle {
    cmd_tx: mpsc::Sender<AgentCmd>,
    current_cancel: Arc<Mutex<CancellationToken>>,
    busy: Arc<std::sync::atomic::AtomicBool>,
    mode: Arc<Mutex<AgentMode>>,
}

impl AgentHandle {
    /// Spawn the agent task. Events flow through the session's event channel.
    pub fn spawn(agent: Agent, factory: Option<AgentFactory>) -> Self {
        let mut agent = agent;
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<AgentCmd>(16);
        let current_cancel = Arc::new(Mutex::new(agent.cancel.clone()));
        let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mode = Arc::new(Mutex::new(AgentMode::Main));
        let cc = current_cancel.clone();
        let busy2 = busy.clone();
        let mode2 = mode.clone();
        tokio::spawn(async move {
            async fn run(agent: &mut Agent, text: String, cc: &Arc<Mutex<CancellationToken>>, busy: &Arc<std::sync::atomic::AtomicBool>) {
                agent.cancel = CancellationToken::new();
                *cc.lock() = agent.cancel.clone();
                busy.store(true, std::sync::atomic::Ordering::SeqCst);
                let session = agent.session.clone();
                let outcome = agent.run_turn(text).await.unwrap_or_else(|e| AgentOutcome::EngineError(format!("{e:#}")));
                busy.store(false, std::sync::atomic::Ordering::SeqCst);
                session.emit_async(AgentEvent::Finished { reason: outcome.as_str().to_string(), turns: agent.turn() }).await;
            }
            while let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    AgentCmd::UserMessage(text) => run(&mut agent, text, &cc, &busy2).await,
                    AgentCmd::SwitchMode { mode: m, first_message } => {
                        let Some(f) = &factory else { agent.session.emit(AgentEvent::Warning { text: "mode switching not available".into() }); continue; };
                        match f(m).await {
                            Ok(new_agent) => {
                                agent = new_agent;
                                *mode2.lock() = m;
                                agent.session.emit(AgentEvent::Warning { text: format!("switched to {m:?} mode (fresh context)") });
                                if let Some(t) = first_message { run(&mut agent, t, &cc, &busy2).await; }
                            }
                            Err(e) => agent.session.emit(AgentEvent::Error { text: format!("could not switch mode: {e:#}") }),
                        }
                    }
                    AgentCmd::SetEffort(e) => agent.effort_override = e,
                    AgentCmd::SetTask(t) => agent.task = t,
                    AgentCmd::Compact => {
                        let session = agent.session.clone();
                        if let Err(e) = agent.compact().await { session.emit(AgentEvent::Warning { text: format!("compaction failed: {e:#}") }); }
                    }
                    AgentCmd::Stats(reply) => {
                        let _ = reply.send(AgentStats {
                            used_tokens: agent.ctx.used_tokens_estimate(),
                            n_ctx: agent.ctx.n_ctx(),
                            compactions: agent.ctx.compactions,
                            messages: agent.ctx.len(),
                            prefix_hash: agent.ctx.prefix().hash_hex(),
                            turn: agent.turn(),
                            slot_switches: agent.session.engine.slot().switch_count(),
                        });
                    }
                    AgentCmd::Shutdown => break,
                }
            }
        });
        Self { cmd_tx, current_cancel, busy, mode }
    }

    pub fn mode(&self) -> AgentMode { *self.mode.lock() }
    /// Clone of the command sender (for background tasks that need to talk to the agent).
    pub fn sender(&self) -> mpsc::Sender<AgentCmd> { self.cmd_tx.clone() }
    pub fn is_busy(&self) -> bool { self.busy.load(std::sync::atomic::Ordering::SeqCst) }
    pub fn abort(&self) { self.current_cancel.lock().cancel(); }
    pub async fn send(&self, cmd: AgentCmd) -> bool { self.cmd_tx.send(cmd).await.is_ok() }
    pub fn try_send(&self, cmd: AgentCmd) -> bool { self.cmd_tx.try_send(cmd).is_ok() }
    pub async fn stats(&self) -> Option<AgentStats> {
        let (tx, rx) = oneshot::channel();
        if !self.send(AgentCmd::Stats(tx)).await { return None; }
        rx.await.ok()
    }
}
