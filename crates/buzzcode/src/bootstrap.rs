//! Build a Session + agents from config. Shared by headless and TUI modes.

use anyhow::Result;
use cb_config::{Config, PermissionModeCfg};
use cb_core::context::{ContextStore, FrozenPrefix};
use cb_core::events::AgentEvent;
use cb_core::handle::{AgentFactory, AgentMode};
use cb_core::permission::{PermissionBroker, PermissionRequest};
use cb_core::plan::{PlanStore, WritePlan};
use cb_core::prompt::{main_system_prompt, plan_system_prompt, PromptEnv};
use cb_core::registry::ToolRegistry;
use cb_core::session::Session;
use cb_core::subagent::{SpawnAgent, SubagentFactory, EXPLORE_TOOLS};
use cb_core::task::TaskClass;
use cb_core::tokens::TokenCounter;
use cb_core::{Agent, Message, PermissionMode};
use cb_engine::EngineManager;
use cb_tool_api::Extensions;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct Built {
    pub session: Arc<Session>,
    pub agent: Agent,
    pub factory: AgentFactory,
    pub plans: Arc<PlanStore>,
    pub events_rx: mpsc::Receiver<AgentEvent>,
    pub perm_rx: Option<mpsc::Receiver<PermissionRequest>>,
}

pub fn permission_mode(cfg: &Config, override_: Option<PermissionMode>) -> PermissionMode {
    override_.unwrap_or(match cfg.general.permission_mode {
        PermissionModeCfg::Ask => PermissionMode::Ask,
        PermissionModeCfg::AutoAcceptEdits => PermissionMode::AutoAcceptEdits,
        PermissionModeCfg::Yolo => PermissionMode::Yolo,
    })
}

/// Shared pieces the factory needs to mint agents.
struct Shared {
    session: Arc<Session>,
    full_registry: Arc<ToolRegistry>,
    env: PromptEnv,
    repo_map: Option<String>,
    codeintel: Option<Arc<cb_codeintel::tools::CodeIntel>>,
}

impl Shared {
    async fn prefix(&self, system: String, registry: &ToolRegistry) -> FrozenPrefix {
        let mut prefix = FrozenPrefix::new(Message::system(system), registry.freeze());
        prefix.system_tokens = self.session.counter.count(&prefix.system.content).await;
        prefix.tools_tokens = self.session.counter.count(&serde_json::to_string(&*prefix.tools_json).unwrap_or_default()).await;
        prefix
    }

    async fn agent(&self, mode: AgentMode) -> Result<Agent> {
        let cfg = &self.session.cfg;
        let (registry, system, task, max_turns, map_budget) = match mode {
            AgentMode::Main => {
                let r = self.full_registry.without(&["write_plan"]);
                let mut env = self.env.clone(); env.tool_names = r.names();
                (r, main_system_prompt(&env), TaskClass::Chat, cfg.general.max_turns, cfg.context.repo_map_tokens)
            }
            AgentMode::Plan => {
                let mut allow: Vec<&str> = EXPLORE_TOOLS.to_vec();
                allow.push("write_plan"); allow.push("spawn_agent");
                let r = self.full_registry.scoped(&allow);
                let mut env = self.env.clone(); env.tool_names = r.names();
                (r, plan_system_prompt(&env), TaskClass::Plan, cfg.general.max_turns, cfg.context.repo_map_tokens_plan)
            }
        };
        let registry = Arc::new(registry);
        let prefix = self.prefix(system, &registry).await;
        tracing::info!(?mode, system_tokens = prefix.system_tokens, tools_tokens = prefix.tools_tokens, hash = %prefix.hash_hex(), "frozen prefix");
        let mut ctx = ContextStore::new(prefix, self.session.engine.n_ctx(), self.session.counter.clone());
        let map = match (&self.codeintel, map_budget) {
            (Some(ci), b) if b != cfg.context.repo_map_tokens => Some(cb_codeintel::tools::initial_map(ci, b)).filter(|m| !m.trim().is_empty()),
            _ => self.repo_map.clone(),
        };
        if let Some(m) = map { ctx.set_repo_map(m); }
        // Warm the KV cache with the full stable prefix (system + tools + repo map), same effort as the session.
        let effort = task.effort(&self.session.engine.profile()).to_string();
        match self.session.engine.warmup(ctx.to_request_messages(), Some((*ctx.tools_json()).clone()), &effort).await {
            Ok(t) => tracing::info!(prompt_n = t.prompt_n, cache_n = t.cache_n, ms = t.prompt_ms, "prefix warmed"),
            Err(e) => tracing::warn!("warmup failed: {e:#}"),
        }
        Ok(Agent::new(self.session.clone(), ctx, registry, task, max_turns))
    }
}

/// `interactive` = wire a permission channel (TUI); headless relies on mode.
pub async fn build(cfg: Arc<Config>, mode: PermissionMode, interactive: bool) -> Result<Built> {
    let profile = cfg.active_profile().clone();
    let engine = EngineManager::new(cfg.clone(), profile);
    engine.start().await?;

    let extensions = Arc::new(Extensions::default());

    // Code intelligence (index + tools), unless disabled.
    let mut codeintel: Option<Arc<cb_codeintel::tools::CodeIntel>> = None;
    if cfg.codeintel.enabled {
        let db = cfg.paths.project_dir.join(&cfg.codeintel.index_path);
        match cb_codeintel::Indexer::open(&cfg.paths.project_dir, &db, &cfg.codeintel) {
            Ok(idx) => {
                let idx = Arc::new(idx);
                match idx.scan() { Ok(st) => tracing::info!(?st, "index ready"), Err(e) => tracing::warn!("index scan failed: {e:#}") }
                let ci = Arc::new(cb_codeintel::tools::CodeIntel::new(idx.clone(), cfg.context.repo_map_tokens));
                extensions.insert(ci.clone());
                let observers = Arc::new(cb_tool_api::EditObservers::default());
                observers.push(Arc::new(IndexObserver { idx: idx.clone(), ci: ci.clone() }));
                extensions.insert(observers);
                codeintel = Some(ci);
            }
            Err(e) => tracing::warn!("code intelligence disabled: {e:#}"),
        }
    }

    let (events_tx, events_rx) = mpsc::channel(16384);
    let (perm_tx, perm_rx) = if interactive { let (t, r) = mpsc::channel(4); (Some(t), Some(r)) } else { (None, None) };
    let perms = PermissionBroker::new(mode, &cfg.permissions.always_allow, &cfg.permissions.always_ask_shell_patterns, perm_tx);
    let counter = Arc::new(TokenCounter::new(Some(engine.client().clone())));

    // Commit-message generator backed by the engine (TaskClass::CommitMsg).
    {
        let engine2 = engine.clone();
        let generator = cb_tools::git::CommitMessageGen(Box::new(move |diff: String| {
            let engine = engine2.clone();
            Box::pin(async move {
                let profile = engine.profile();
                let s = TaskClass::CommitMsg.sampling(&profile);
                let msgs = vec![
                    serde_json::json!({"role":"system","content": cb_core::prompt::commit_message_prompt()}),
                    serde_json::json!({"role":"user","content": diff}),
                ];
                let mut req = cb_engine::client::ChatRequest::new(&engine.alias(), msgs);
                req.max_tokens = Some(200);
                req.temperature = s.temperature; req.top_p = s.top_p; req.top_k = s.top_k; req.presence_penalty = s.presence_penalty;
                req = req.reasoning_effort(TaskClass::CommitMsg.effort(&profile));
                let _g = engine.slot().acquire(u64::MAX).await;
                engine.client().chat_once(req).await.ok().map(|(c, _, _, _)| c.trim().trim_matches('`').trim().to_string()).filter(|s| !s.is_empty())
            })
        }));
        extensions.insert(Arc::new(generator));
    }

    let session = Arc::new(Session {
        cfg: cfg.clone(), engine: engine.clone(), registry: Arc::new(ToolRegistry::new()), perms, counter: counter.clone(),
        events: events_tx, project_dir: cfg.paths.project_dir.clone(), extensions,
        id: session_id(),
    });

    // Full registry: builtin + code intel + plan + subagents (subagent factory needs the env + registry → two-phase).
    let plans = Arc::new(PlanStore::default());
    let mut base = ToolRegistry::new();
    for t in cb_tools::builtin_tools() { base.register(t)?; }
    if codeintel.is_some() { for t in cb_codeintel::tools::tools() { base.register(t)?; } }
    base.register(Arc::new(WritePlan { store: plans.clone(), plans_dir: cfg.paths.plans_dir() }))?;
    // MCP servers must be connected BEFORE the registry is frozen (tools JSON is part of the stable prefix).
    for t in cb_mcp::connect_all(&cfg.mcp_servers).await { base.register(t)?; }
    let base_names = base.names();

    let languages = match &codeintel {
        Some(ci) => ci.indexer.language_summary().into_iter().take(3).map(|(l, _)| l).collect(),
        None => detect_languages(&cfg.paths.project_dir),
    };
    let repo_map = codeintel.as_ref().map(|ci| cb_codeintel::tools::initial_map(ci, cfg.context.repo_map_tokens)).filter(|m| !m.trim().is_empty());
    let env = PromptEnv::detect(&cfg.paths.project_dir, base_names, languages, repo_map.is_some());

    let sub_factory = Arc::new(SubagentFactory { session: session.clone(), full_registry: Arc::new(base.without(&["write_plan"])), env: env.clone() });
    base.register(Arc::new(SpawnAgent { factory: sub_factory }))?;
    let full_registry = Arc::new(base);

    let shared = Arc::new(Shared { session: session.clone(), full_registry, env, repo_map, codeintel });
    let agent = shared.agent(AgentMode::Main).await?;
    let factory: AgentFactory = {
        let shared = shared.clone();
        Arc::new(move |mode| { let s = shared.clone(); Box::pin(async move { s.agent(mode).await }) })
    };
    Ok(Built { session, agent, factory, plans, events_rx, perm_rx })
}

fn session_id() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{t:x}")
}

struct IndexObserver { idx: Arc<cb_codeintel::Indexer>, ci: Arc<cb_codeintel::tools::CodeIntel> }
impl cb_tool_api::EditObserver for IndexObserver {
    fn on_edits(&self, paths: &[std::path::PathBuf]) {
        if let Err(e) = self.idx.update_paths(paths) { tracing::warn!("index update failed: {e:#}"); }
        self.ci.invalidate_map();
    }
}

/// Top languages by file count (cheap walk, depth-limited) for the system prompt.
pub fn detect_languages(root: &std::path::Path) -> Vec<String> {
    let mut counts: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    let mut wb = ignore::WalkBuilder::new(root);
    wb.hidden(true).git_ignore(true).max_depth(Some(6));
    let mut n = 0;
    for e in wb.build().flatten() {
        if !e.file_type().map(|t| t.is_file()).unwrap_or(false) { continue; }
        n += 1; if n > 20_000 { break; }
        let ext = e.path().extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
        let lang = match ext.as_str() {
            "rs" => "Rust", "py" => "Python", "ts" | "tsx" => "TypeScript", "js" | "jsx" | "mjs" => "JavaScript", "go" => "Go",
            "c" | "h" => "C", "cpp" | "cc" | "hpp" | "cxx" => "C++", "java" => "Java", "cs" => "C#", "rb" => "Ruby", "php" => "PHP",
            "swift" => "Swift", "kt" => "Kotlin", "sh" | "ps1" => "Shell",
            _ => continue,
        };
        *counts.entry(lang).or_default() += 1;
    }
    let mut v: Vec<(&str, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.into_iter().take(3).map(|(l, _)| l.to_string()).collect()
}
