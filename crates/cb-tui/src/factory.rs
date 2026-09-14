//! Agent "factory" state: every agent is a worker with a real task and a live activity.
//! Shown as a simple text panel in the TUI and as the graphical BUZZCODE ARCADE page (arcade.rs).

use cb_core::events::AgentEvent;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState { Idle, Thinking, Writing, Tool, Waiting, Done, Failed }

#[derive(Debug, Clone)]
pub struct Worker {
    pub id: u64,
    pub name: String,
    pub role: String,
    pub task: String,
    pub state: WorkerState,
    pub activity: String,
    pub tools_used: u32,
    pub turns: u32,
    pub started: Instant,
    pub finished: Option<Instant>,
    /// glyphs of tools used, newest last (the pickups collected)
    pub loot: Vec<char>,
    pub errors: u32,
}

#[derive(Debug, Clone)]
pub struct ToolRecord {
    pub id: String,
    pub name: String,
    pub detail: String,
    pub glyph: char,
    pub is_running: bool,
    pub is_error: bool,
    pub started: Instant,
    pub duration_ms: Option<u128>,
}

pub struct Factory {
    pub workers: Vec<Worker>,
    /// id of the player currently holding the controls (0 = P1 / main agent)
    pub active: u64,
    pub big_task: String,
    pub hi_score: u64,
    pub tools: Vec<ToolRecord>,
}

fn new_worker(id: u64, name: &str, role: &str, task: &str) -> Worker {
    Worker { id, name: name.into(), role: role.into(), task: task.lines().next().unwrap_or("").to_string(), state: WorkerState::Idle, activity: "press start".into(), tools_used: 0, turns: 0, started: Instant::now(), finished: None, loot: Vec::new(), errors: 0 }
}

impl Default for Factory {
    fn default() -> Self { Self { workers: vec![new_worker(0, "FOREMAN", "main", "")], active: 0, big_task: String::new(), hi_score: 0, tools: Vec::new() } }
}

impl Factory {
    fn active_mut(&mut self) -> &mut Worker {
        let id = self.active;
        if let Some(i) = self.workers.iter().position(|w| w.id == id) { &mut self.workers[i] } else { &mut self.workers[0] }
    }

    pub fn set_big_task(&mut self, t: &str) {
        self.hi_score = self.hi_score.max(self.score());
        self.big_task = t.lines().next().unwrap_or("").to_string();
        self.workers.truncate(1);
        let w = &mut self.workers[0];
        w.task = self.big_task.clone(); w.state = WorkerState::Thinking; w.activity = "reading the mission".into(); w.started = Instant::now(); w.finished = None; w.tools_used = 0; w.turns = 0; w.loot.clear(); w.errors = 0;
        self.tools.clear();
        self.active = 0;
    }

    /// 100 per tool pickup, 250 per level (turn), 1000 stage-clear bonus per finished player, −50 per error.
    pub fn score(&self) -> u64 {
        self.workers.iter().map(|w| {
            let base = w.tools_used as u64 * 100 + w.turns as u64 * 250 + if w.state == WorkerState::Done { 1000 } else { 0 };
            base.saturating_sub(w.errors as u64 * 50)
        }).sum()
    }

    pub fn apply(&mut self, ev: &AgentEvent) {
        match ev {
            AgentEvent::TurnStart { turn, .. } => { let w = self.active_mut(); w.turns = *turn; if w.state != WorkerState::Done { w.state = WorkerState::Thinking; w.activity = "thinking…".into(); } }
            AgentEvent::ReasoningDelta { .. } => { let w = self.active_mut(); w.state = WorkerState::Thinking; if !w.activity.starts_with("thinking") { w.activity = "thinking…".into(); } }
            AgentEvent::ContentDelta { .. } => { let w = self.active_mut(); w.state = WorkerState::Writing; w.activity = "writing the answer".into(); }
            AgentEvent::ToolCallStart { id, name, arguments, .. } => {
                let w = self.active_mut();
                w.state = WorkerState::Tool;
                w.tools_used += 1;
                w.loot.push(glyph(name));
                if w.loot.len() > 40 { w.loot.remove(0); }
                let desc = describe(name, arguments);
                w.activity = desc.clone();
                let rec = ToolRecord {
                    id: id.clone(),
                    name: name.clone(),
                    detail: desc,
                    glyph: glyph(name),
                    is_running: true,
                    is_error: false,
                    started: Instant::now(),
                    duration_ms: None,
                };
                self.tools.push(rec);
                if self.tools.len() > 10 { self.tools.remove(0); }
            }
            AgentEvent::ToolCallResult { id, name, is_error, .. } => {
                let w = self.active_mut();
                if *is_error { w.errors += 1; w.activity = format!("{name} failed — retrying"); }
                if let Some(rec) = self.tools.iter_mut().rev().find(|t| t.id == *id || t.is_running) {
                    rec.is_running = false;
                    rec.is_error = *is_error;
                    rec.duration_ms = Some(rec.started.elapsed().as_millis());
                }
            }
            AgentEvent::PermissionRequested { name, .. } => { let w = self.active_mut(); w.state = WorkerState::Waiting; w.activity = format!("waiting for permission: {name}"); }
            AgentEvent::Verify { command, ok, .. } => { let w = self.active_mut(); w.state = WorkerState::Tool; if !*ok { w.errors += 1; } w.activity = format!("{} {}", if *ok { "✓ checked" } else { "✗ check failed" }, short(command, 26)); }
            AgentEvent::AgentSpawned { id, kind, task } => {
                let n = self.workers.len();
                { let f = self.active_mut(); f.state = WorkerState::Waiting; f.activity = format!("waiting for P{}", n + 1); }
                let mut w = new_worker(*id, player_name(n), kind, task);
                w.state = WorkerState::Thinking; w.activity = "reading the brief".into();
                self.workers.push(w);
                self.active = *id;
            }
            AgentEvent::AgentFinished { id, outcome, turns } => {
                if let Some(w) = self.workers.iter_mut().find(|w| w.id == *id) {
                    w.state = if outcome == "end_turn" { WorkerState::Done } else { WorkerState::Failed };
                    w.turns = *turns; w.finished = Some(Instant::now());
                    w.activity = if outcome == "end_turn" { "STAGE CLEAR — report delivered".into() } else { format!("stopped: {outcome}") };
                }
                self.active = 0;
                let f = &mut self.workers[0]; f.state = WorkerState::Thinking; f.activity = "reading the report".into();
            }
            AgentEvent::Finished { reason, turns } => {
                let f = &mut self.workers[0];
                f.turns = *turns; f.finished = Some(Instant::now());
                f.state = if reason == "end_turn" { WorkerState::Done } else if reason == "model_switch" { WorkerState::Idle } else { WorkerState::Failed };
                f.activity = match reason.as_str() { "end_turn" => "STAGE CLEAR".into(), "model_switch" => "changing cabinet".into(), r => format!("GAME OVER: {r}") };
                self.active = 0;
                self.hi_score = self.hi_score.max(self.score());
            }
            AgentEvent::Compaction { .. } => { let w = self.active_mut(); w.activity = "power-up: context compacted".into(); }
            _ => {}
        }
    }
}

fn player_name(n: usize) -> &'static str {
    const NAMES: &[&str] = &["SCOUT", "BOLT", "PIXEL", "WRENCH", "BYTE", "GEAR", "SPARK", "NOVA", "RIVET", "QUILL"];
    NAMES[(n.saturating_sub(1)) % NAMES.len()]
}

fn short(s: &str, n: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= n { s } else { format!("{}…", s.chars().take(n.saturating_sub(1)).collect::<String>()) }
}

fn arg<'a>(v: &'a serde_json::Value, k: &str) -> &'a str { v.get(k).and_then(|x| x.as_str()).unwrap_or("") }

/// Pickup glyph per tool.
fn glyph(name: &str) -> char {
    match name {
        "read_file" | "outline" => '▤', "edit_file" | "write_file" => '◆', "grep" | "glob" | "symbol_search" | "list_dir" | "repo_map" => '◎',
        "shell" => '⚙', "spawn_agent" => '★', "write_plan" => '▣', n if n.starts_with("git_") => '♦', _ => '●',
    }
}

pub fn describe(name: &str, a: &serde_json::Value) -> String {
    match name {
        "read_file" => format!("reading {}", short(arg(a, "path"), 28)),
        "edit_file" => format!("editing {}", short(arg(a, "path"), 28)),
        "write_file" => format!("writing {}", short(arg(a, "path"), 28)),
        "grep" => format!("searching '{}'", short(arg(a, "pattern"), 20)),
        "glob" => format!("finding {}", short(arg(a, "pattern"), 22)),
        "list_dir" => "browsing folders".into(),
        "shell" => format!("running `{}`", short(arg(a, "command"), 24)),
        "outline" => format!("skimming {}", short(arg(a, "path"), 28)),
        "symbol_search" => format!("looking up '{}'", short(arg(a, "query"), 20)),
        "repo_map" => "studying the map".into(),
        "spawn_agent" => format!("calling a {} player", arg(a, "kind")),
        "write_plan" => "drawing the plan".into(),
        "git_commit" => "committing".into(),
        n if n.starts_with("git_") => format!("git {}", &n[4..]),
        n if n.starts_with("mcp__") => format!("using {}", n.trim_start_matches("mcp__").replace("__", "/")),
        n => format!("using {n}"),
    }
}

/// Three-line stick-figure sprite for the text panel; `f` animates it.
fn sprite(state: WorkerState, f: bool) -> [&'static str; 3] {
    match state {
        WorkerState::Idle => ["  o  ", r" /|\ ", r" / \ "],
        WorkerState::Thinking => if f { [" o ? ", r" /|\ ", r" / \ "] } else { [" o . ", r" /|\ ", r" / \ "] },
        WorkerState::Writing => if f { ["  o  ", " /|✎ ", r" / \ "] } else { ["  o  ", " /|_ ", r" / \ "] },
        WorkerState::Tool => if f { [" o ⚒ ", " /|/ ", r" / \ "] } else { ["  o  ", " /|⚒ ", r" / \ "] },
        WorkerState::Waiting => if f { ["  o  ", r" \|/ ", r" / \ "] } else { ["  o  ", r" /|\ ", r" / \ "] },
        WorkerState::Done => ["  o ☕", r" /|\ ", r" / \ "],
        WorkerState::Failed => ["  x  ", r" /|\ ", r" / \ "],
    }
}

fn color(state: WorkerState) -> Color {
    match state { WorkerState::Idle => Color::DarkGray, WorkerState::Thinking => Color::Cyan, WorkerState::Writing => Color::LightBlue, WorkerState::Tool => Color::Yellow, WorkerState::Waiting => Color::Magenta, WorkerState::Done => Color::Green, WorkerState::Failed => Color::Red }
}

fn title_case(s: &str) -> String {
    let mut c = s.chars();
    match c.next() { Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(), None => String::new() }
}

/// The "factory floor" side panel showing subagents and live tools with progress.
pub fn draw(f: &mut Frame, area: Rect, factory: &Factory, frame: usize) {
    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)).title(" [factory floor] ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let w = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();
    let task = if factory.big_task.is_empty() { "no job yet — type a task".to_string() } else { factory.big_task.clone() };
    lines.push(Line::from(vec![
        Span::styled("MISSION", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
    ]));
    for piece in textwrap::wrap(&task, w.max(8)).into_iter().take(2) {
        lines.push(Line::from(Span::styled(piece.into_owned(), Style::default().fg(Color::White))));
    }
    lines.push(Line::from(Span::styled("─".repeat(w), Style::default().fg(Color::DarkGray))));

    // 1. SUBAGENTS & WORKERS SECTION
    lines.push(Line::from(vec![
        Span::styled("AGENTS & WORKERS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" ({})", factory.workers.len()), Style::default().fg(Color::DarkGray)),
    ]));

    for wk in &factory.workers {
        let c = color(wk.state);
        let sp = sprite(wk.state, (frame + wk.id as usize) % 2 == 0);
        let active = wk.id == factory.active && !matches!(wk.state, WorkerState::Done | WorkerState::Failed | WorkerState::Idle);
        let state_badge = match wk.state {
            WorkerState::Idle => "[IDLE]",
            WorkerState::Thinking => "[THINKING]",
            WorkerState::Writing => "[WRITING]",
            WorkerState::Tool => "[RUNNING TOOL]",
            WorkerState::Waiting => "[WAITING]",
            WorkerState::Done => "[DONE ✓]",
            WorkerState::Failed => "[FAILED ✗]",
        };
        let title = format!("{}{} · {} {}", if active { "▶ " } else { "" }, title_case(&wk.name), wk.role, state_badge);
        let elapsed = wk.finished.unwrap_or_else(Instant::now).duration_since(wk.started).as_secs();
        let meta = format!("turn {} · {} tools · {}s", wk.turns, wk.tools_used, elapsed);
        let text_w = w.saturating_sub(8);
        lines.push(Line::from(vec![
            Span::styled(sp[0].to_string(), Style::default().fg(c)),
            Span::raw("  "),
            Span::styled(short(&title, text_w), Style::default().fg(c).add_modifier(Modifier::BOLD))
        ]));
        lines.push(Line::from(vec![
            Span::styled(sp[1].to_string(), Style::default().fg(c)),
            Span::raw("  "),
            Span::styled(short(&wk.activity, text_w), Style::default().fg(Color::White))
        ]));
        lines.push(Line::from(vec![
            Span::styled(sp[2].to_string(), Style::default().fg(c)),
            Span::raw("  "),
            Span::styled(short(&meta, text_w), Style::default().fg(Color::DarkGray))
        ]));
        if !wk.loot.is_empty() {
            let loot_str: String = wk.loot.iter().rev().take(text_w.min(18)).collect();
            lines.push(Line::from(vec![
                Span::raw("       "),
                Span::styled(format!("tools: {loot_str}"), Style::default().fg(Color::LightYellow))
            ]));
        }
        if wk.id != 0 && !wk.task.is_empty() {
            for piece in textwrap::wrap(&wk.task, text_w.max(8)).into_iter().take(1) {
                lines.push(Line::from(vec![Span::raw("       "), Span::styled(piece.into_owned(), Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC))]));
            }
        }
        lines.push(Line::from(""));
    }

    // 2. TOOLS IN USE SECTION
    lines.push(Line::from(Span::styled("─".repeat(w), Style::default().fg(Color::DarkGray))));
    lines.push(Line::from(vec![
        Span::styled("🛠 TOOLS IN USE", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
    ]));

    if factory.tools.is_empty() {
        lines.push(Line::from(Span::styled("  no tools invoked yet", Style::default().fg(Color::DarkGray))));
    } else {
        for t in factory.tools.iter().rev().take(6) {
            let (status_icon, status_style) = if t.is_running {
                ("⏳ running…", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
            } else if t.is_error {
                ("✗ error", Style::default().fg(Color::Red))
            } else {
                ("✓ done", Style::default().fg(Color::Green))
            };
            let dur_str = match t.duration_ms {
                Some(ms) => format!(" ({}ms)", ms),
                None => format!(" ({}s)", t.started.elapsed().as_secs()),
            };
            let tool_line = format!("{} {} {}{}", t.glyph, t.name, status_icon, dur_str);
            lines.push(Line::from(Span::styled(short(&tool_line, w), status_style)));
            if !t.detail.is_empty() {
                lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::styled(short(&t.detail, w.saturating_sub(4)), Style::default().fg(Color::DarkGray))
                ]));
            }
        }
    }

    let h = inner.height as usize;
    let visible: Vec<Line> = lines.into_iter().take(h).collect();
    f.render_widget(Paragraph::new(visible), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn player_lifecycle_and_score() {
        let mut f = Factory::default();
        f.set_big_task("Build the river game");
        f.apply(&AgentEvent::TurnStart { turn: 0, task: cb_core::TaskClass::Chat, effort: "medium".into() });
        f.apply(&AgentEvent::ToolCallStart { id: "1".into(), name: "spawn_agent".into(), arguments: json!({"kind":"explore","task":"find the canvas code"}) });
        f.apply(&AgentEvent::AgentSpawned { id: 7, kind: "explore".into(), task: "find the canvas code".into() });
        assert_eq!(f.active, 7);
        assert_eq!(f.workers[1].name, "SCOUT");
        assert_eq!(f.tools.len(), 1);
        assert!(f.tools[0].is_running);

        f.apply(&AgentEvent::ToolCallStart { id: "2".into(), name: "grep".into(), arguments: json!({"pattern":"canvas"}) });
        assert_eq!(f.workers[1].activity, "searching 'canvas'");
        assert_eq!(f.workers[1].loot, vec!['◎']);
        assert_eq!(f.tools.len(), 2);

        f.apply(&AgentEvent::ToolCallResult { id: "2".into(), name: "grep".into(), output: "found".into(), is_error: false, truncated: false, elapsed_ms: 120 });
        assert!(!f.tools[1].is_running);
        assert!(!f.tools[1].is_error);

        f.apply(&AgentEvent::AgentFinished { id: 7, outcome: "end_turn".into(), turns: 3 });
        assert_eq!(f.workers[1].state, WorkerState::Done);
        assert_eq!(f.active, 0);
        // P1: 1 tool; P2: 1 tool + 3 turns + clear bonus
        assert_eq!(f.score(), 100 + 100 + 750 + 1000);
        f.apply(&AgentEvent::Finished { reason: "end_turn".into(), turns: 5 });
        assert_eq!(f.workers[0].state, WorkerState::Done);
        assert!(f.hi_score >= f.score());
    }
}
