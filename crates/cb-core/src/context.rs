//! Prefix-stable conversation store.
//!
//! Invariants:
//! * `prefix` (system message + tools JSON) is serialized once and never changes for the
//!   lifetime of the store — its blake3 hash is recorded so tests/bench can assert stability.
//! * `log` is append-only; the only other mutation is `compact()`, which is an explicit,
//!   logged cache break.
//! * Every request serializes exactly `[prefix.system] ++ log` with fixed key order.

use crate::message::{Message, Role};
use crate::tokens::TokenCounter;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct FrozenPrefix {
    pub system: Message,
    pub tools_json: Arc<Value>,
    pub bytes_hash: [u8; 32],
    pub system_tokens: u32,
    pub tools_tokens: u32,
}

impl FrozenPrefix {
    pub fn new(system: Message, tools_json: Arc<Value>) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(serde_json::to_string(&system.to_value()).unwrap_or_default().as_bytes());
        h.update(b"\0tools\0");
        h.update(serde_json::to_string(&*tools_json).unwrap_or_default().as_bytes());
        Self { system, tools_json, bytes_hash: *h.finalize().as_bytes(), system_tokens: 0, tools_tokens: 0 }
    }
    pub fn hash_hex(&self) -> String { blake3::Hash::from(self.bytes_hash).to_hex().to_string() }
    pub fn tokens(&self) -> u32 { self.system_tokens + self.tools_tokens }
}

/// Per-message template overhead (role tags etc.), a small constant.
const MSG_OVERHEAD: u32 = 6;

pub struct ContextStore {
    pub id: u64,
    prefix: FrozenPrefix,
    log: Vec<Message>,
    n_ctx: u32,
    counter: Arc<TokenCounter>,
    /// Index into `log` where the latest compaction summary lives (None = never compacted).
    summary_at: Option<usize>,
    /// Number of compactions performed (each is a cache break).
    pub compactions: u32,
    /// Last server-reported total prompt tokens (prompt_n + cache_n); ground truth for usage.
    pub last_prompt_total: Option<u32>,
}

impl ContextStore {
    pub fn new(prefix: FrozenPrefix, n_ctx: u32, counter: Arc<TokenCounter>) -> Self {
        Self { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), prefix, log: Vec::new(), n_ctx, counter, summary_at: None, compactions: 0, last_prompt_total: None }
    }

    pub fn prefix(&self) -> &FrozenPrefix { &self.prefix }
    pub fn n_ctx(&self) -> u32 { self.n_ctx }
    pub fn set_n_ctx(&mut self, n: u32) { self.n_ctx = n; }
    pub fn log(&self) -> &[Message] { &self.log }
    pub fn len(&self) -> usize { self.log.len() }
    pub fn is_empty(&self) -> bool { self.log.is_empty() }
    pub fn last(&self) -> Option<&Message> { self.log.last() }
    pub fn counter(&self) -> &Arc<TokenCounter> { &self.counter }

    /// The ONLY way to add to the conversation.
    pub fn push(&mut self, m: Message) -> &Message {
        self.log.push(m);
        self.log.last().unwrap()
    }

    /// Exact messages array for the request. Prefix system message first, then the log.
    pub fn to_request_messages(&self) -> Vec<Value> {
        let mut v = Vec::with_capacity(self.log.len() + 1);
        v.push(self.prefix.system.to_value());
        v.extend(self.log.iter().map(Message::to_value));
        v
    }

    pub fn tools_json(&self) -> Arc<Value> { self.prefix.tools_json.clone() }

    /// Stable hash of the *entire* request messages (for cache-bench prefix checks).
    pub fn request_hash(&self) -> [u8; 32] {
        let s = serde_json::to_string(&self.to_request_messages()).unwrap_or_default();
        *blake3::hash(s.as_bytes()).as_bytes()
    }

    /// Heuristic token usage (sync). Prefers the server-reported total when available and then
    /// adds estimates for messages appended since.
    pub fn used_tokens_estimate(&self) -> u32 {
        let mut total = self.prefix.tokens();
        for m in &self.log {
            total += m.tokens.get().unwrap_or_else(|| self.counter.estimate(&m.content) + (m.tool_calls.len() as u32) * 24) + MSG_OVERHEAD;
        }
        total
    }

    /// Exact-ish usage: count uncounted messages via the server.
    pub async fn used_tokens(&self) -> u32 {
        let mut total = self.prefix.tokens();
        for m in &self.log {
            let t = match m.tokens.get() {
                Some(t) => t,
                None => {
                    let mut t = self.counter.count(&m.content).await;
                    for c in &m.tool_calls { t += self.counter.count(&serde_json::to_string(&c.arguments).unwrap_or_default()).await + 8; }
                    m.tokens.set(Some(t));
                    t
                }
            };
            total += t + MSG_OVERHEAD;
        }
        total
    }

    /// Record the server-reported prompt size after a request; refresh the learned ratio.
    pub fn observe_prompt_total(&mut self, total: u32) {
        self.last_prompt_total = Some(total);
        let chars: usize = self.prefix.system.chars() + self.log.iter().map(Message::chars).sum::<usize>();
        self.counter.calibrate(total, chars);
    }

    pub fn used_tokens_now(&self) -> u32 {
        self.last_prompt_total.map(|t| t.max(self.used_tokens_estimate())).unwrap_or_else(|| self.used_tokens_estimate())
    }

    pub fn needs_compaction(&self, threshold: f32, output_reserve: u32) -> bool {
        let used = self.used_tokens_now();
        let limit = ((self.n_ctx as f32) * threshold) as u32;
        used >= limit || (used + output_reserve > self.n_ctx)
    }

    /// Whether the context window is full (100% capacity reached).
    pub fn is_full(&self) -> bool {
        let used = self.used_tokens_now();
        used >= self.n_ctx
    }

    /// Restart the conversation when context becomes full.
    /// Preserves the frozen prefix, preserves the repository map (if present),
    /// extracts the original user goal/task, and starts a fresh context with that goal.
    pub fn restart_with_goal(&mut self) -> String {
        let original_goal = self.log.iter()
            .find(|m| m.role == Role::User && !m.meta.is_repo_map && !m.meta.is_summary)
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "Continue previous task".to_string());

        let repo_map = self.log.iter().find(|m| m.meta.is_repo_map).cloned();
        self.log.clear();
        if let Some(rm) = repo_map { self.log.push(rm); }

        let restart_note = format!(
            "[Context reached capacity — restarted with fresh context]\nInitial task:\n{original_goal}\n\nPlease proceed with this task from where you left off."
        );
        let mut msg = Message::user(restart_note);
        msg.meta.is_summary = true;
        self.log.push(msg);

        self.summary_at = Some(self.log.len() - 1);
        self.compactions += 1;
        self.last_prompt_total = None;
        original_goal
    }

    /// Replace the log with `[summary] ++ tail`, where `tail` is the last `keep_tail_turns`
    /// complete user turns (cut on a user message so no orphan tool results remain).
    pub fn compact(&mut self, summary: String, keep_tail_turns: usize) -> CompactionReport {
        let before = self.log.len();
        let mut cut = self.log.len();
        let mut seen = 0;
        for (i, m) in self.log.iter().enumerate().rev() {
            if m.role == Role::User && !m.meta.is_summary && !m.meta.is_repo_map {
                seen += 1;
                if seen >= keep_tail_turns { cut = i; break; }
            }
        }
        if seen < keep_tail_turns { cut = self.log.iter().position(|m| m.role == Role::User && !m.meta.is_repo_map).unwrap_or(0); }
        let tail: Vec<Message> = self.log.drain(cut..).collect();
        let repo_map = self.log.iter().find(|m| m.meta.is_repo_map).cloned();
        self.log.clear();
        if let Some(rm) = repo_map { self.log.push(rm); }
        let mut s = Message::user(format!("[Conversation summary — earlier turns were compacted]\n{summary}"));
        s.meta.is_summary = true;
        self.summary_at = Some(self.log.len());
        self.log.push(s);
        self.log.extend(tail);
        self.compactions += 1;
        self.last_prompt_total = None;
        CompactionReport { messages_before: before, messages_after: self.log.len() }
    }

    /// Replace or insert the repo-map message (message #1). Allowed only when the cache is
    /// already being broken (compaction / new session) — callers enforce that.
    pub fn set_repo_map(&mut self, content: String) {
        let mut m = Message::developer_or_user(content);
        m.meta.is_repo_map = true;
        if let Some(pos) = self.log.iter().position(|x| x.meta.is_repo_map) { self.log[pos] = m; }
        else { self.log.insert(0, m); }
    }

    /// New store for a subagent: a different frozen prefix, empty log.
    pub fn fork(&self, prefix: FrozenPrefix) -> ContextStore {
        ContextStore::new(prefix, self.n_ctx, self.counter.clone())
    }

    /// Drop trailing assistant message(s) that have tool calls without results (after abort).
    pub fn repair_tail(&mut self) {
        while let Some(last) = self.log.last() {
            if last.role == Role::Assistant && !last.tool_calls.is_empty() {
                let ids: Vec<&str> = last.tool_calls.iter().map(|c| c.id.as_str()).collect();
                let _ = ids;
                self.log.pop();
            } else { break; }
        }
    }
}

impl Message {
    /// Repo map / injected context prefers the `developer` role when supported; we use `user`
    /// with a fixed header because llama.cpp templates for Qwen map developer→system anyway.
    pub fn developer_or_user(content: String) -> Message {
        Message::user(format!("[Repository map]\n{content}"))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompactionReport { pub messages_before: usize, pub messages_after: usize }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store() -> ContextStore {
        let prefix = FrozenPrefix::new(Message::system("sys"), Arc::new(json!([])));
        ContextStore::new(prefix, 32768, Arc::new(TokenCounter::new(None)))
    }

    #[test]
    fn prefix_bytes_stable_across_pushes() {
        let mut s = store();
        let h0 = s.prefix().bytes_hash;
        let r0 = serde_json::to_string(&s.to_request_messages()).unwrap();
        s.push(Message::user("hello"));
        let r1 = serde_json::to_string(&s.to_request_messages()).unwrap();
        assert!(r1.starts_with(r0.trim_end_matches(']')), "prefix bytes changed");
        assert_eq!(h0, s.prefix().bytes_hash);
    }

    #[test]
    fn compaction_keeps_tail_turns_and_system() {
        let mut s = store();
        for i in 0..5 {
            s.push(Message::user(format!("u{i}")));
            s.push(Message::assistant(format!("a{i}"), vec![]));
        }
        let sys_before = s.to_request_messages()[0].clone();
        s.compact("summary".into(), 2);
        let msgs = s.to_request_messages();
        assert_eq!(msgs[0], sys_before);
        assert!(s.log()[0].meta.is_summary);
        assert_eq!(s.log().len(), 1 + 4); // summary + 2 user turns × (user+assistant)
        assert_eq!(s.log()[1].content, "u3");
    }

    #[test]
    fn compaction_triggers_at_95_percent() {
        let mut s = store();
        // 95% of 32768 is 31129
        s.observe_prompt_total(31128);
        assert!(!s.needs_compaction(0.95, 0));
        assert!(!s.is_full());

        s.observe_prompt_total(31130);
        assert!(s.needs_compaction(0.95, 0));
        assert!(!s.is_full());

        s.observe_prompt_total(32768);
        assert!(s.needs_compaction(0.95, 0));
        assert!(s.is_full());
    }

    #[test]
    fn full_context_restart_preserves_initial_goal() {
        let mut s = store();
        s.push(Message::user("Build a web server in Rust"));
        s.push(Message::assistant("I will create main.rs", vec![]));
        s.push(Message::user("Also add logging"));
        s.observe_prompt_total(32768);
        assert!(s.is_full());

        let goal = s.restart_with_goal();
        assert_eq!(goal, "Build a web server in Rust");
        assert_eq!(s.log().len(), 1);
        assert!(s.log()[0].content.contains("Build a web server in Rust"));
        assert!(!s.is_full());
        assert_eq!(s.compactions, 1);
    }
}
