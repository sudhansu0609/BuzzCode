//! Terminal UI for buzzcode (ratatui + crossterm).

mod app;
mod arcade;
mod blocks;
mod factory;
mod input;
mod ui;

pub use app::{run, TuiOptions};

/// Run the graphical arcade page with a simulated crew (no engine). `buzzcode arcade --demo`.
pub async fn arcade_demo(port: u16, open: bool) -> anyhow::Result<()> {
    use cb_core::events::AgentEvent;
    use serde_json::json;
    let server = arcade::ArcadeServer::start(port).await?;
    println!("BUZZCODE ARCADE demo at {}  (Ctrl+C to stop)", server.url());
    if open { server.open_browser(); }
    let mut f = factory::Factory::default();
    let script: Vec<AgentEvent> = vec![
        AgentEvent::TurnStart { turn: 0, task: cb_core::TaskClass::Chat, effort: "medium".into() },
        AgentEvent::ReasoningDelta { text: "…".into() },
        AgentEvent::ToolCallStart { id: "1".into(), name: "repo_map".into(), arguments: json!({}) },
        AgentEvent::ToolCallStart { id: "2".into(), name: "read_file".into(), arguments: json!({"path":"index.html"}) },
        AgentEvent::ToolCallStart { id: "3".into(), name: "spawn_agent".into(), arguments: json!({"kind":"explore","task":"find the canvas game loop and collision code"}) },
        AgentEvent::AgentSpawned { id: 7, kind: "explore".into(), task: "find the canvas game loop and collision code".into() },
        AgentEvent::TurnStart { turn: 0, task: cb_core::TaskClass::Explore, effort: "medium".into() },
        AgentEvent::ToolCallStart { id: "4".into(), name: "grep".into(), arguments: json!({"pattern":"requestAnimationFrame"}) },
        AgentEvent::ToolCallStart { id: "5".into(), name: "read_file".into(), arguments: json!({"path":"game.js"}) },
        AgentEvent::ToolCallStart { id: "6".into(), name: "symbol_search".into(), arguments: json!({"query":"collide"}) },
        AgentEvent::ContentDelta { text: "FINDINGS".into() },
        AgentEvent::AgentFinished { id: 7, outcome: "end_turn".into(), turns: 3 },
        AgentEvent::TurnStart { turn: 1, task: cb_core::TaskClass::Chat, effort: "medium".into() },
        AgentEvent::ToolCallStart { id: "7".into(), name: "edit_file".into(), arguments: json!({"path":"game.js"}) },
        AgentEvent::Verify { command: "npx tsc --noEmit".into(), ok: false, output: String::new() },
        AgentEvent::ToolCallResult { id: "7".into(), name: "edit_file".into(), output: String::new(), is_error: true, truncated: false, elapsed_ms: 0 },
        AgentEvent::ToolCallStart { id: "8".into(), name: "edit_file".into(), arguments: json!({"path":"game.js"}) },
        AgentEvent::Verify { command: "npx tsc --noEmit".into(), ok: true, output: String::new() },
        AgentEvent::ToolCallStart { id: "9".into(), name: "shell".into(), arguments: json!({"command":"npm test"}) },
        AgentEvent::ContentDelta { text: "Done".into() },
        AgentEvent::Finished { reason: "end_turn".into(), turns: 4 },
    ];
    loop {
        f.set_big_task("Build an HTML5 canvas river game: rowing man, obstacles, score, game over");
        server.publish(arcade::snapshot_json(&f, "qwen3.8-27b (demo)", 44.5, 5400, 65536, true));
        for ev in &script {
            tokio::time::sleep(std::time::Duration::from_millis(if matches!(ev, AgentEvent::ToolCallStart { .. }) { 1400 } else { 700 })).await;
            f.apply(ev);
            let busy = !matches!(ev, AgentEvent::Finished { .. });
            server.publish(arcade::snapshot_json(&f, "qwen3.8-27b (demo)", 44.5 + (f.score() % 7) as f64, 5400 + f.score() as u32, 65536, busy));
        }
        tokio::time::sleep(std::time::Duration::from_secs(6)).await;
    }
}
