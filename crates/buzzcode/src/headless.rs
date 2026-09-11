//! `buzzcode -p "prompt"`: run one task without a TUI. `--json` prints events as JSON lines.

use crate::bootstrap;
use anyhow::Result;
use cb_config::Config;
use cb_core::events::AgentEvent;
use cb_core::PermissionMode;
use std::io::Write;
use std::sync::Arc;

pub async fn run(cfg: Arc<Config>, prompt: String, json: bool, mode: Option<PermissionMode>) -> Result<()> {
    let mode = mode.unwrap_or(bootstrap::permission_mode(&cfg, None));
    let built = bootstrap::build(cfg.clone(), mode, false).await?;
    let bootstrap::Built { session, mut agent, mut events_rx, factory, plans, perm_rx } = built;
    // These hold Arc<Session> (and thus the event sender); drop them so the printer can finish.
    drop(factory); drop(plans); drop(perm_rx);

    let printer = tokio::spawn(async move {
        let mut in_reasoning = false;
        while let Some(ev) = events_rx.recv().await {
            let mut out = std::io::stdout();
            if json {
                let _ = writeln!(out, "{}", serde_json::to_string(&ev).unwrap_or_default());
                continue;
            }
            match ev {
                AgentEvent::ReasoningDelta { text } => { if !in_reasoning { let _ = write!(out, "\x1b[2m"); in_reasoning = true; } let _ = write!(out, "{text}"); }
                AgentEvent::ContentDelta { text } => { if in_reasoning { let _ = write!(out, "\x1b[0m\n"); in_reasoning = false; } let _ = write!(out, "{text}"); }
                AgentEvent::AssistantMessage { .. } => { if in_reasoning { let _ = write!(out, "\x1b[0m"); in_reasoning = false; } let _ = writeln!(out); }
                AgentEvent::ToolCallStart { name, arguments, .. } => { let _ = writeln!(out, "\x1b[36m▸ {name} {}\x1b[0m", compact(&arguments)); }
                AgentEvent::ToolCallResult { name, output, is_error, .. } => {
                    let first: String = output.lines().take(8).collect::<Vec<_>>().join("\n  ");
                    let more = output.lines().count().saturating_sub(8);
                    let _ = writeln!(out, "  \x1b[{}m{name} → {first}{}\x1b[0m", if is_error { "31" } else { "90" }, if more > 0 { format!("\n  … {more} more lines") } else { String::new() });
                }
                AgentEvent::Verify { command, ok, .. } => { let _ = writeln!(out, "\x1b[35m⚙ verify `{command}` → {}\x1b[0m", if ok { "ok" } else { "FAILED" }); }
                AgentEvent::Metrics { timings, ctx_used, ctx_total, cache_ratio, decode_tps } => {
                    let _ = writeln!(out, "\x1b[90m[ctx {ctx_used}/{ctx_total} · {decode_tps:.1} tok/s · cache {:.0}% · prompt {:.0} ms{}]\x1b[0m", cache_ratio * 100.0, timings.prompt_ms, timings.draft_acceptance().map(|a| format!(" · mtp {:.0}%", a * 100.0)).unwrap_or_default());
                }
                AgentEvent::Warning { text } => { let _ = writeln!(out, "\x1b[33m! {text}\x1b[0m"); }
                AgentEvent::Error { text } => { let _ = writeln!(out, "\x1b[31m✗ {text}\x1b[0m"); }
                AgentEvent::Compaction { messages_before, messages_after, summary_tokens } => { let _ = writeln!(out, "\x1b[33m↻ compacted {messages_before}→{messages_after} messages ({summary_tokens} summary tokens)\x1b[0m"); }
                AgentEvent::Finished { reason, turns } => { let _ = writeln!(out, "\x1b[90m— finished ({reason}, {turns} turns)\x1b[0m"); }
                AgentEvent::AgentSpawned { id, kind, task } => { let _ = writeln!(out, "\x1b[35m▸ worker #{id} ({kind}) started: {}\x1b[0m", task.lines().next().unwrap_or("").chars().take(120).collect::<String>()); }
                AgentEvent::AgentFinished { id, outcome, turns } => { let _ = writeln!(out, "\x1b[35m▸ worker #{id} finished ({outcome}, {turns} turns)\x1b[0m"); }
                _ => {}
            }
            let _ = out.flush();
        }
    });

    let ctrlc = tokio::signal::ctrl_c();
    let cancel = agent.cancel.clone();
    tokio::spawn(async move { let _ = ctrlc.await; cancel.cancel(); });

    let outcome = agent.run_turn(prompt).await?;
    session.emit(AgentEvent::Finished { reason: outcome.as_str().to_string(), turns: agent.turn() });
    let engine = session.engine.clone();
    drop(agent);
    drop(session);
    // The registry's spawn_agent tool also holds a Session; give the printer a bounded wait.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), printer).await;
    engine.stop().await;
    Ok(())
}

fn compact(v: &serde_json::Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_default();
    if s.len() > 160 { format!("{}…", &s[..160]) } else { s }
}
