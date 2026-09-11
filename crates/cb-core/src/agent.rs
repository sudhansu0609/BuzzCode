//! The agent loop: generate → parse → permission → execute → verify → repeat.

use crate::context::ContextStore;
use crate::events::AgentEvent;
use crate::guard::{GuardError, LoopGuard};
use crate::message::{Message, Role, ToolCall};
use crate::parser::{ParseOutcome, ToolCallParser};
use crate::permission::PermissionDecision;
use crate::registry::ToolRegistry;
use crate::session::Session;
use crate::task::TaskClass;
use crate::truncate;
use anyhow::{Context, Result};
use cb_engine::client::{ChatRequest, StreamEvent};
use cb_engine::sse::ToolCallAcc;
use cb_tool_api::{EditRecord, OutputBudget, PermissionClass, ToolCtx, ToolError, ToolOutput};
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentOutcome {
    /// Model produced a final answer (no tool calls).
    EndTurn,
    MaxTurns,
    Aborted,
    LoopDetected(String),
    EngineError(String),
}

impl AgentOutcome {
    pub fn as_str(&self) -> &'static str {
        match self { Self::EndTurn => "end_turn", Self::MaxTurns => "max_turns", Self::Aborted => "aborted", Self::LoopDetected(_) => "loop_detected", Self::EngineError(_) => "engine_error" }
    }
}

/// Escalating recovery for malformed tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RetryLevel { Normal, Reminded, Forced }

pub struct Agent {
    pub session: Arc<Session>,
    pub ctx: ContextStore,
    pub registry: Arc<ToolRegistry>,
    parser: ToolCallParser,
    guard: LoopGuard,
    pub task: TaskClass,
    pub cancel: CancellationToken,
    retry: RetryLevel,
    /// Effort override (`/effort`), else from task class.
    pub effort_override: Option<String>,
    /// When true, tool results are not displayed in events (subagents).
    pub quiet: bool,
    edits_since_verify: u32,
}

impl Agent {
    pub fn new(session: Arc<Session>, ctx: ContextStore, registry: Arc<ToolRegistry>, task: TaskClass, max_turns: u32) -> Self {
        let parser = ToolCallParser::new(registry.names());
        Self {
            session, ctx, registry, parser, guard: LoopGuard::new(max_turns), task,
            cancel: CancellationToken::new(), retry: RetryLevel::Normal, effort_override: None, quiet: false, edits_since_verify: 0,
        }
    }

    fn effort(&self) -> String {
        self.effort_override.clone().unwrap_or_else(|| self.task.effort(&self.session.engine.profile()).to_string())
    }

    pub fn build_request(&self, tool_choice: Option<Value>, max_tokens: Option<u32>) -> ChatRequest {
        let profile = self.session.engine.profile();
        let s = self.task.sampling(&profile);
        let mut req = ChatRequest::new(&self.session.engine.alias(), self.ctx.to_request_messages());
        if !self.registry.is_empty() { req.tools = Some((*self.ctx.tools_json()).clone()); }
        req.tool_choice = tool_choice;
        req.max_tokens = Some(max_tokens.unwrap_or(profile.max_tokens));
        req.temperature = s.temperature; req.top_p = s.top_p; req.top_k = s.top_k; req.min_p = s.min_p;
        req.presence_penalty = s.presence_penalty; req.repeat_penalty = s.repeat_penalty;
        req = req.reasoning_effort(&self.effort());
        req
    }

    /// Run one user turn to completion.
    pub async fn run_turn(&mut self, user_input: String) -> Result<AgentOutcome> {
        self.ctx.push(Message::user(user_input));
        self.guard.new_user_message();
        self.run_loop().await
    }

    /// Continue the loop without a new user message (e.g. after a denied permission).
    pub async fn run_loop(&mut self) -> Result<AgentOutcome> {
        let cfg = self.session.cfg.clone();
        loop {
            if self.cancel.is_cancelled() { self.ctx.repair_tail(); return Ok(AgentOutcome::Aborted); }

            // 1. If context is completely full, start again with a fresh context.
            if self.ctx.is_full() {
                self.start_again();
            }

            // 2. If context reached compaction threshold (95%), compact before it becomes full.
            if self.ctx.needs_compaction(cfg.context.compaction_threshold, cfg.context.output_reserve_tokens) {
                if let Err(e) = self.compact().await { self.session.emit(AgentEvent::Warning { text: format!("compaction failed: {e:#}") }); }
                if self.ctx.is_full() {
                    self.start_again();
                }
            }

            self.session.emit(AgentEvent::TurnStart { turn: self.guard.turn, task: self.task, effort: self.effort() });
            let tool_choice = if self.retry == RetryLevel::Forced { Some(json!("required")) } else { None };
            let req = self.build_request(tool_choice, None);

            let (mut content, mut reasoning, native, timings) = match self.stream_assistant(req).await {
                Ok(v) => v,
                Err(e) => {
                    if self.cancel.is_cancelled() { self.ctx.repair_tail(); return Ok(AgentOutcome::Aborted); }
                    let err_text = format!("{e:#}");
                    if is_context_overflow(&err_text) || self.ctx.is_full() {
                        self.session.emit(AgentEvent::Warning { text: format!("engine context full ({err_text}); restarting with fresh context") });
                        self.start_again();
                        continue;
                    }
                    self.session.emit(AgentEvent::Error { text: err_text.clone() });
                    return Ok(AgentOutcome::EngineError(err_text));
                }
            };
            if reasoning.is_empty() {
                for (tag_open, tag_close) in [("<think>", "</think>"), ("<thought>", "</thought>")] {
                    if let Some(start) = content.find(tag_open) {
                        if let Some(end) = content.find(tag_close) {
                            let think_text = &content[start + tag_open.len()..end];
                            reasoning = think_text.trim().to_string();
                            content = format!("{}{}", &content[..start], &content[end + tag_close.len()..]).trim().to_string();
                        } else {
                            let think_text = &content[start + tag_open.len()..];
                            reasoning = think_text.trim().to_string();
                            content = content[..start].trim().to_string();
                        }
                        break;
                    }
                }
            }
            if let Some(t) = &timings {
                self.ctx.observe_prompt_total(t.prompt_n + t.cache_n + t.predicted_n);
                let used = t.prompt_n + t.cache_n + t.predicted_n;
                self.session.emit(AgentEvent::Metrics { timings: t.clone(), ctx_used: used, ctx_total: self.ctx.n_ctx(), cache_ratio: t.cache_ratio(), decode_tps: t.decode_tps() });
            }

            let outcome = self.parser.parse(&native, &content);
            match outcome {
                ParseOutcome::NoCalls => {
                    if let Err(e) = self.guard.observe_text(&content) {
                        return Ok(self.finish_guard(e));
                    }
                    let mut m = Message::assistant(content.clone(), vec![]);
                    m.meta.reasoning = if reasoning.is_empty() { None } else { Some(reasoning) };
                    m.meta.turn = self.guard.turn;
                    self.ctx.push(m);
                    self.session.emit(AgentEvent::AssistantMessage { content, tool_calls: vec![] });
                    self.retry = RetryLevel::Normal;
                    return Ok(AgentOutcome::EndTurn);
                }
                ParseOutcome::Malformed { error, snippet } => {
                    tracing::warn!(%error, "malformed tool call");
                    // Keep the assistant text so the model sees its own mistake, then correct it.
                    let mut m = Message::assistant(content.clone(), vec![]);
                    m.meta.turn = self.guard.turn;
                    self.ctx.push(m);
                    let hint = match self.retry {
                        RetryLevel::Normal => { self.retry = RetryLevel::Reminded; format!("Your tool call could not be parsed: {error}. Snippet: {snippet}\nRespond again using the native tool-call format with valid JSON arguments. Available tools: {}.", self.registry.names().join(", ")) }
                        _ => { self.retry = RetryLevel::Forced; format!("Tool call still malformed: {error}. You MUST emit a well-formed tool call now.") }
                    };
                    self.ctx.push(Message::user(hint));
                    self.session.emit(AgentEvent::Warning { text: format!("malformed tool call ({error}); retrying") });
                    if let Err(e) = self.guard.next_turn() { return Ok(self.finish_guard(e)); }
                    continue;
                }
                ParseOutcome::Calls(pc) => {
                    self.retry = RetryLevel::Normal;
                    let mut m = Message::assistant(pc.leftover_text.clone(), pc.calls.clone());
                    m.meta.reasoning = if reasoning.is_empty() { None } else { Some(reasoning) };
                    m.meta.turn = self.guard.turn;
                    self.ctx.push(m);
                    self.session.emit(AgentEvent::AssistantMessage { content: pc.leftover_text.clone(), tool_calls: pc.calls.clone() });

                    if let Err(e) = self.guard.observe_calls(&pc.calls) {
                        if matches!(e, GuardError::RepeatedCall(_, n) if n >= 4) {
                            return Ok(self.finish_guard(e));
                        }
                        // Give the model one explicit chance to change course, then stop.
                        let text = format!("{e}. Stop repeating this call. Either use a different approach or explain to the user why you are stuck.");
                        for c in &pc.calls { self.ctx.push(Message::tool(&c.id, &c.name, &text, true)); }
                        self.session.emit(AgentEvent::Warning { text: e.to_string() });
                        if let Err(e2) = self.guard.next_turn() { return Ok(self.finish_guard(e2)); }
                        continue;
                    }

                    let results = self.execute_calls(pc.calls).await;
                    let mut any_error = false;
                    let mut edits: Vec<EditRecord> = Vec::new();
                    for (call, out) in results {
                        any_error |= out.is_error;
                        if out.is_error {
                            if let Err(e) = self.guard.observe_error(&call.name, &out.text) {
                                return Ok(self.finish_guard(e));
                            }
                        }
                        edits.extend(out.edits.iter().cloned());
                        let mut msg = Message::tool(&call.id, &call.name, out.text, out.is_error);
                        msg.meta.truncated = out.truncated.is_some();
                        msg.meta.spill_path = out.truncated.as_ref().map(|t| t.spill_path.clone());
                        msg.meta.turn = self.guard.turn;
                        self.ctx.push(msg);
                    }
                    if !edits.is_empty() { self.maybe_verify(&edits).await; }
                    let _ = any_error;
                    if let Err(e) = self.guard.next_turn() { return Ok(self.finish_guard(e)); }
                }
            }
        }
    }

    fn finish_guard(&mut self, e: GuardError) -> AgentOutcome {
        self.session.emit(AgentEvent::Warning { text: e.to_string() });
        match e { GuardError::MaxTurns(_) => AgentOutcome::MaxTurns, other => AgentOutcome::LoopDetected(other.to_string()) }
    }

    // ------------------------------------------------------------------ streaming

    async fn stream_assistant(&mut self, req: ChatRequest) -> Result<(String, String, Vec<ToolCallAcc>, Option<cb_engine::Timings>)> {
        let engine = self.session.engine.clone();
        let guard = engine.slot().acquire(self.ctx.id).await;
        if guard.switched { tracing::debug!(prev = ?guard.previous_owner, "slot owner switched; expect prompt cache miss"); }
        let mut stream = engine.client().chat_stream(&req).await.context("starting generation")?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<ToolCallAcc> = Vec::new();
        let mut timings = None;
        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => { anyhow::bail!("cancelled"); }
                ev = stream.next() => {
                    let Some(ev) = ev else { break };
                    match ev {
                        StreamEvent::Content(t) => { if !self.quiet { self.session.emit(AgentEvent::ContentDelta { text: t.clone() }); } content.push_str(&t); }
                        StreamEvent::Reasoning(t) => { if !self.quiet { self.session.emit(AgentEvent::ReasoningDelta { text: t.clone() }); } reasoning.push_str(&t); }
                        StreamEvent::ToolCallDelta { index, id, name, args_fragment } => {
                            let acc = match calls.iter_mut().find(|c| c.index == index) {
                                Some(c) => c,
                                None => { calls.push(ToolCallAcc { index, ..Default::default() }); calls.last_mut().unwrap() }
                            };
                            if let Some(id) = id { acc.id = id; }
                            if let Some(n) = name { acc.name.push_str(&n); }
                            acc.arguments.push_str(&args_fragment);
                        }
                        StreamEvent::Timings(t) => timings = Some(t),
                        StreamEvent::Error(e) => anyhow::bail!("engine: {e}"),
                        StreamEvent::Done(_) | StreamEvent::Usage { .. } => {}
                    }
                }
            }
        }
        drop(guard);
        Ok((content, reasoning, calls, timings))
    }

    // ------------------------------------------------------------------ execution

    fn tool_ctx(&self) -> ToolCtx {
        let c = &self.session.cfg.context;
        ToolCtx {
            cwd: self.session.project_dir.clone(),
            project_dir: self.session.project_dir.clone(),
            spill_dir: self.session.spill_dir(),
            cancel: self.cancel.child_token(),
            budget: OutputBudget { max_bytes: c.tool_output_max_bytes, head_lines: c.tool_output_head_lines, tail_lines: c.tool_output_tail_lines },
            permission_mode: self.session.perms.mode(),
            turn: self.guard.turn,
            extensions: self.session.extensions.clone(),
        }
    }

    /// Validate → permission → run. Read-only calls run concurrently; others sequentially in order.
    async fn execute_calls(&mut self, calls: Vec<ToolCall>) -> Vec<(ToolCall, ToolOutput)> {
        let cx = self.tool_ctx();
        let mut results: Vec<Option<(ToolCall, ToolOutput)>> = (0..calls.len()).map(|_| None).collect();
        let mut ro_tasks = Vec::new();
        let mut serial: Vec<(usize, ToolCall)> = Vec::new();

        for (i, call) in calls.into_iter().enumerate() {
            self.session.emit(AgentEvent::ToolCallStart { id: call.id.clone(), name: call.name.clone(), arguments: call.arguments.clone() });
            let Some(tool) = self.registry.get(&call.name) else {
                results[i] = Some((call.clone(), ToolOutput::error(format!("unknown tool `{}`; available: {}", call.name, self.registry.names().join(", ")))));
                continue;
            };
            if let Err(e) = self.registry.validate(&call.name, &call.arguments) {
                let schema = serde_json::to_string(&tool.spec().schema).unwrap_or_default();
                results[i] = Some((call.clone(), ToolOutput::error(format!("{e}. Schema for `{}`: {schema}", call.name))));
                continue;
            }
            let class = tool.spec().class;
            // Permission (with preview for writes).
            let needs = self.session.perms.needs_prompt(&call, class);
            if needs {
                let preview = match tool.preview(&call.arguments, &cx).await { Ok(p) => p, Err(e) => Some(format!("(preview failed: {e})")) };
                self.session.emit(AgentEvent::PermissionRequested { id: call.id.clone(), name: call.name.clone(), preview: preview.clone() });
                let d = self.session.perms.resolve(&call, class, preview).await;
                let allowed = d != PermissionDecision::Deny;
                self.session.emit(AgentEvent::PermissionResolved { id: call.id.clone(), allowed });
                if !allowed {
                    results[i] = Some((call.clone(), ToolOutput::error("Denied by user. Do not retry this exact action; ask the user or choose another approach.")));
                    continue;
                }
            }
            if class == PermissionClass::ReadOnly {
                let cx = cx.clone();
                let call2 = call.clone();
                ro_tasks.push((i, tokio::spawn(async move { let r = run_tool(tool, &call2, &cx).await; (call2, r) })));
            } else {
                serial.push((i, call));
            }
        }
        for (i, h) in ro_tasks {
            match h.await {
                Ok((call, out)) => { self.emit_result(&call, &out); results[i] = Some((call, out)); }
                Err(e) => { results[i] = Some((ToolCall { id: format!("join_{i}"), name: "?".into(), arguments: Value::Null }, ToolOutput::error(format!("tool task panicked: {e}")))); }
            }
        }
        for (i, call) in serial {
            let tool = self.registry.get(&call.name).unwrap();
            let out = run_tool(tool, &call, &cx).await;
            self.emit_result(&call, &out);
            results[i] = Some((call, out));
        }
        results.into_iter().flatten().collect()
    }

    fn emit_result(&self, call: &ToolCall, out: &ToolOutput) {
        self.session.emit(AgentEvent::ToolCallResult {
            id: call.id.clone(), name: call.name.clone(),
            output: if self.quiet { String::new() } else { out.text.clone() },
            is_error: out.is_error, truncated: out.truncated.is_some(), elapsed_ms: 0,
        });
    }

    // ------------------------------------------------------------------ verify loop

    /// After edits: run configured verify commands (rate-limited) and append the output as a
    /// user-visible tool result attached to a synthetic call (append-only, cache-safe).
    async fn maybe_verify(&mut self, edits: &[EditRecord]) {
        // Keep the code index fresh.
        if let Some(obs) = self.session.extensions.get::<cb_tool_api::EditObservers>() {
            let paths: Vec<std::path::PathBuf> = edits.iter().map(|e| e.path.clone()).collect();
            obs.notify(&paths);
        }
        self.edits_since_verify += edits.len() as u32;
        if self.edits_since_verify < 1 { return; }
        self.edits_since_verify = 0;
        let cfg = self.session.cfg.clone();
        let mut commands: Vec<String> = Vec::new();
        for e in edits {
            let name = e.path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            for (glob, cmds) in &cfg.verify {
                if glob_match(glob, &name) {
                    for c in cmds {
                        let c = c.replace("{file}", &e.path.to_string_lossy().replace('\\', "/"));
                        if !commands.contains(&c) { commands.push(c); }
                    }
                }
            }
        }
        if commands.is_empty() { return; }
        let mut report = String::new();
        let mut any_fail = false;
        for c in commands.iter().take(3) {
            let (ok, out) = run_shell_capture(&self.session.project_dir, c, Duration::from_secs(90)).await;
            any_fail |= !ok;
            let tail: Vec<&str> = out.lines().rev().take(80).collect::<Vec<_>>().into_iter().rev().collect();
            report.push_str(&format!("$ {c}\n[{}]\n{}\n", if ok { "ok" } else { "FAILED" }, tail.join("\n")));
            self.session.emit(AgentEvent::Verify { command: c.clone(), ok, output: out.clone() });
        }
        // Attach to the conversation as an assistant tool call + result pair so the format stays valid.
        let call = ToolCall { id: format!("verify_{}", self.guard.turn), name: "shell".into(), arguments: json!({"command": commands.join(" ; "), "_auto_verify": true}) };
        let mut a = Message::assistant("", vec![call.clone()]);
        a.meta.turn = self.guard.turn;
        self.ctx.push(a);
        let text = if any_fail { format!("Automatic verification after your edits FAILED. Fix these before finishing:\n{report}") } else { format!("Automatic verification after your edits passed.\n{report}") };
        let mut t = Message::tool(&call.id, "shell", text, any_fail);
        t.meta.turn = self.guard.turn;
        self.ctx.push(t);
    }

    // ------------------------------------------------------------------ compaction

    pub async fn compact(&mut self) -> Result<()> {
        let cfg = self.session.cfg.clone();
        let max = cfg.context.summary_max_tokens;
        let prompt = crate::prompt::summarize_prompt(max - 300);
        let mut messages = self.ctx.to_request_messages();
        messages.push(json!({"role": "user", "content": prompt}));
        let profile = self.session.engine.profile();
        let s = TaskClass::Summarize.sampling(&profile);
        let mut req = ChatRequest::new(&self.session.engine.alias(), messages);
        req.max_tokens = Some(max);
        req.temperature = s.temperature; req.top_p = s.top_p; req.top_k = s.top_k; req.presence_penalty = s.presence_penalty;
        req = req.reasoning_effort(TaskClass::Summarize.effort(&profile));
        let engine = self.session.engine.clone();
        let summary = {
            let _g = engine.slot().acquire(self.ctx.id).await;
            let (content, _r, _t, _) = engine.client().chat_once(req).await.context("summarizing for compaction")?;
            content
        };
        let summary_tokens = self.session.counter.estimate(&summary);
        let rep = self.ctx.compact(summary, cfg.context.keep_tail_turns);
        self.session.emit(AgentEvent::Compaction { messages_before: rep.messages_before, messages_after: rep.messages_after, summary_tokens });
        Ok(())
    }

    pub fn abort(&self) { self.cancel.cancel(); }
    pub fn turn(&self) -> u32 { self.guard.turn }

    pub fn transcript(&self) -> &[Message] { self.ctx.log() }
    pub fn last_assistant_text(&self) -> Option<&str> {
        self.ctx.log().iter().rev().find(|m| m.role == Role::Assistant && !m.content.is_empty()).map(|m| m.content.as_str())
    }

    /// Restart the session when context reaches capacity.
    /// Resets conversation history to the original user goal with a fresh context.
    pub fn start_again(&mut self) {
        let goal = self.ctx.restart_with_goal();
        self.guard = LoopGuard::new(self.guard.max_turns);
        self.retry = RetryLevel::Normal;
        self.session.emit(AgentEvent::Warning {
            text: format!("context reached capacity: restarted session with fresh context for task \"{goal}\""),
        });
    }
}

fn is_context_overflow(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    (lower.contains("context") && (lower.contains("exceed") || lower.contains("full") || lower.contains("too long") || lower.contains("overflow") || lower.contains("length")))
        || lower.contains("n_ctx")
}

async fn run_tool(tool: Arc<dyn cb_tool_api::Tool>, call: &ToolCall, cx: &ToolCtx) -> ToolOutput {
    let t0 = Instant::now();
    let res = tokio::time::timeout(Duration::from_secs(600), tool.call(call.arguments.clone(), cx)).await;
    let mut out = match res {
        Ok(Ok(o)) => o,
        Ok(Err(ToolError::Cancelled)) => ToolOutput::error("cancelled"),
        Ok(Err(e)) => ToolOutput::error(e.to_string()),
        Err(_) => ToolOutput::error("tool timed out after 600s"),
    };
    // Uniform truncation.
    let stem = format!("t{}-{}-{}", cx.turn, call.name, &call.id);
    let tr = truncate::apply(&out.text, &cx.budget, &cx.spill_dir, &stem);
    out.text = tr.text;
    if tr.info.is_some() { out.truncated = tr.info; }
    tracing::debug!(tool = %call.name, ms = t0.elapsed().as_millis(), error = out.is_error, "tool done");
    out
}

/// Minimal glob: supports `*` and `?` and `**/`-less patterns (filename level).
pub fn glob_match(pat: &str, name: &str) -> bool {
    fn rec(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => rec(&p[1..], s) || (!s.is_empty() && rec(p, &s[1..])),
            (Some(b'?'), Some(_)) => rec(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a.eq_ignore_ascii_case(b) => rec(&p[1..], &s[1..]),
            _ => false,
        }
    }
    rec(pat.as_bytes(), name.as_bytes())
}

/// Run a shell command, returning (success, merged output).
pub async fn run_shell_capture(cwd: &std::path::Path, command: &str, timeout: Duration) -> (bool, String) {
    let mut cmd = if cfg!(windows) {
        let mut c = tokio::process::Command::new("powershell");
        c.args(["-NoProfile", "-NonInteractive", "-Command", command]);
        c
    } else {
        let mut c = tokio::process::Command::new("sh");
        c.args(["-c", command]);
        c
    };
    cmd.current_dir(cwd).stdin(std::process::Stdio::null()).kill_on_drop(true);
    let fut = cmd.output();
    match tokio::time::timeout(timeout, fut).await {
        Ok(Ok(o)) => {
            let mut s = String::from_utf8_lossy(&o.stdout).to_string();
            let e = String::from_utf8_lossy(&o.stderr);
            if !e.trim().is_empty() { if !s.is_empty() { s.push('\n'); } s.push_str(&e); }
            (o.status.success(), s)
        }
        Ok(Err(e)) => (false, format!("failed to run: {e}")),
        Err(_) => (false, format!("timed out after {}s", timeout.as_secs())),
    }
}
