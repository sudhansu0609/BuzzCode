//! Drawing.

use crate::app::{App, Overlay};
use cb_engine::EngineState;
use ratatui::prelude::*;
use ratatui::widgets::{Block as WBlock, Borders, Clear, Paragraph, Wrap};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let input_h = (app.input.line_count() as u16).clamp(1, 10) + 2;
    // transcript / input / status bar (bottom row)
    let chunks = Layout::vertical([Constraint::Min(3), Constraint::Length(input_h), Constraint::Length(1)]).split(area);
    if app.show_factory && area.width >= 90 {
        let cols = Layout::horizontal([Constraint::Min(50), Constraint::Length(38)]).split(chunks[0]);
        draw_transcript(f, app, cols[0]);
        crate::factory::draw(f, cols[1], &app.factory, app.spinner / 8);
    } else {
        draw_transcript(f, app, chunks[0]);
    }
    draw_input(f, app, chunks[1]);
    draw_status(f, app, chunks[2]);
    match app.overlay {
        Overlay::Permission => draw_permission(f, app, area),
        Overlay::Help => draw_help(f, area),
        Overlay::EngineLog => draw_engine_log(f, app, area),
        Overlay::None => {}
    }
}

fn draw_transcript(f: &mut Frame, app: &mut App, area: Rect) {
    let width = area.width.saturating_sub(1);
    let show = app.show_reasoning;
    let h = area.height as usize;
    if app.follow { app.scroll_from_bottom = 0; }
    // Gather from the newest block backwards until the viewport (+ scroll offset) is filled —
    // avoids cloning the whole transcript every frame during long generations.
    let need = h + app.scroll_from_bottom;
    let mut chunks: Vec<Vec<Line>> = Vec::new();
    let mut got = 0usize;
    for b in app.blocks.iter_mut().rev() {
        let l = b.lines(width, show);
        got += l.len();
        chunks.push(l.to_vec());
        if got >= need { break; }
    }
    let mut all: Vec<Line> = Vec::with_capacity(got);
    for c in chunks.into_iter().rev() { all.extend(c); }
    let total = all.len();
    let max_from_bottom = total.saturating_sub(h);
    if app.scroll_from_bottom > max_from_bottom { app.scroll_from_bottom = max_from_bottom; }
    let end = total.saturating_sub(app.scroll_from_bottom);
    let start = end.saturating_sub(h);
    let visible: Vec<Line> = all[start..end].to_vec();
    f.render_widget(Paragraph::new(visible), area);
    if app.scroll_from_bottom > 0 {
        let tag = format!(" ↓ {} more lines (PgDn / Ctrl+End) ", app.scroll_from_bottom);
        let w = tag.len() as u16;
        let r = Rect { x: area.right().saturating_sub(w + 1), y: area.bottom().saturating_sub(1), width: w, height: 1 };
        f.render_widget(Paragraph::new(tag).style(Style::default().bg(Color::DarkGray).fg(Color::White)), r);
    }
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let s = &app.stats;
    let pct = 100.0 * s.ctx_used as f64 / s.ctx_total.max(1) as f64;
    let ctx_color = if pct > 75.0 { Color::Red } else if pct > 50.0 { Color::Yellow } else { Color::Green };
    let engine = match &app.engine_state {
        EngineState::Ready { .. } | EngineState::External => Span::styled("●", Style::default().fg(Color::Green)),
        EngineState::Crashed { .. } => Span::styled("● crashed", Style::default().fg(Color::Red)),
        EngineState::Stopped => Span::styled("○ stopped", Style::default().fg(Color::DarkGray)),
        other => Span::styled(format!("◐ {other:?}"), Style::default().fg(Color::Yellow)),
    };
    let busy = if app.busy { format!(" {} ", SPINNER[app.spinner % SPINNER.len()]) } else { "   ".into() };
    let mode = match app.session.perms.mode() { cb_core::PermissionMode::Yolo => "allow-all", cb_core::PermissionMode::Ask => "manual", cb_core::PermissionMode::AutoAcceptEdits => "auto-edits" };
    // Speed: live rate while generating, otherwise the last server-measured rate; session average in parens.
    let tps_span = match (app.busy, s.live_tps()) {
        (true, Some(t)) => Span::styled(format!("⚡ {t:.1} tok/s"), Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
        (true, None) if s.gen_start.is_none() => Span::styled(format!("⏳ prefill… {}", s.turn_start.map(|t| format!("{:.1}s", t.elapsed().as_secs_f64())).unwrap_or_default()), Style::default().fg(Color::Yellow)),
        (true, None) => Span::styled("⚡ … tok/s", Style::default().fg(Color::LightGreen)),
        (false, _) if s.decode_tps > 0.0 => Span::styled(format!("⚡ {:.1} tok/s", s.decode_tps), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        _ => Span::styled("⚡ – tok/s", Style::default().fg(Color::DarkGray)),
    };
    let mut spans = vec![
        Span::styled(busy, Style::default().fg(Color::Cyan)),
        engine, Span::raw(" "),
        Span::styled(app.session.engine.alias(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw(" │ "),
        tps_span,
    ];
    if let Some(avg) = s.session_avg_tps() { spans.push(Span::styled(format!(" (avg {avg:.0})"), Style::default().fg(Color::DarkGray))); }
    spans.push(Span::raw(" │ ctx "));
    spans.push(Span::styled(format!("{}/{} ({pct:.0}%)", fmt_k(s.ctx_used), fmt_k(s.ctx_total)), Style::default().fg(ctx_color)));
    spans.push(Span::raw(format!(" │ cache {:.0}% │ prompt {:.0}ms", s.cache_ratio * 100.0, s.prompt_ms)));
    if let Some(m) = s.mtp { spans.push(Span::raw(format!(" │ mtp {:.0}%", m * 100.0))); }
    let effort_display = if s.effort == "xhigh" { "high" } else { &s.effort };
    spans.push(Span::raw(format!(" │ {mode} │ effort {effort_display} │ turn {}/{}", s.turn, s.max_turns)));
    if let Some((msg, _)) = &app.flash { spans.push(Span::styled(format!("  ◆ {msg}"), Style::default().fg(Color::Magenta))); }
    f.render_widget(Paragraph::new(Line::from(spans)).style(Style::default().bg(Color::Rgb(30, 30, 40))), area);
}

fn draw_input(f: &mut Frame, app: &mut App, area: Rect) {
    let title = if app.busy { format!(" {} thinking… ", SPINNER[app.spinner % SPINNER.len()]) } else { " prompt (Enter send · Shift/Alt/Ctrl+Enter or \\ newline · /help) ".into() };
    let color = if app.busy { Color::Cyan } else { Color::Green };
    let block = WBlock::default().borders(Borders::ALL).border_style(Style::default().fg(color)).title(title);
    app.input.render(f, area, block);
}

fn centered(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
    let v = Layout::vertical([Constraint::Percentage((100 - pct_h) / 2), Constraint::Percentage(pct_h), Constraint::Percentage((100 - pct_h) / 2)]).split(area);
    Layout::horizontal([Constraint::Percentage((100 - pct_w) / 2), Constraint::Percentage(pct_w), Constraint::Percentage((100 - pct_w) / 2)]).split(v[1])[1]
}

fn draw_permission(f: &mut Frame, app: &App, area: Rect) {
    let Some(p) = &app.pending else { return };
    let r = centered(area, 84, 70);
    f.render_widget(Clear, r);
    let args = serde_json::to_string_pretty(&p.req.call.arguments).unwrap_or_default();
    let mut text: Vec<Line> = vec![
        Line::from(vec![Span::styled("Tool: ", Style::default().fg(Color::DarkGray)), Span::styled(p.req.call.name.clone(), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), Span::styled(format!("   ({:?})", p.req.class), Style::default().fg(Color::DarkGray))]),
        Line::from(""),
    ];
    for l in args.lines().take(30) { text.push(Line::from(Span::styled(l.to_string(), Style::default().fg(Color::Gray)))); }
    if let Some(prev) = &p.req.preview {
        text.push(Line::from(""));
        text.push(Line::from(Span::styled("─ preview ─", Style::default().fg(Color::DarkGray))));
        for l in prev.lines() {
            let st = if l.starts_with('+') && !l.starts_with("+++") { Style::default().fg(Color::Green) } else if l.starts_with('-') && !l.starts_with("---") { Style::default().fg(Color::Red) } else if l.starts_with("@@") { Style::default().fg(Color::Cyan) } else { Style::default().fg(Color::Gray) };
            text.push(Line::from(Span::styled(l.to_string(), st)));
        }
    }
    let queued = if app.perm_queue.is_empty() { String::new() } else { format!(" (+{} queued)", app.perm_queue.len()) };
    let block = WBlock::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Yellow))
        .title(format!(" Permission required{queued} "))
        .title_bottom(Line::from(" [y] allow  [a] always allow this tool  [A] allow all (yolo)  [n] deny  ↑↓ scroll ").centered());
    f.render_widget(Paragraph::new(text).block(block).wrap(Wrap { trim: false }).scroll((p.scroll, 0)), r);
}

fn draw_help(f: &mut Frame, area: Rect) {
    let r = centered(area, 70, 70);
    f.render_widget(Clear, r);
    let lines = [
        "Keys",
        "  Enter            send     Shift/Alt/Ctrl+Enter or '\\' newline",
        "  Ctrl+C           abort generation (twice to quit)   Ctrl+D quit",
        "  PgUp/PgDn        scroll   Ctrl+End  jump to bottom",
        "  Ctrl+T           toggle reasoning display",
        "  Ctrl+O           expand/collapse last tool output",
        "  F1 help   F2 engine log   F3 factory panel   ↑/↓ input history",
        "",
        "Commands",
        "  /plan [task]            plan mode: read-only investigation → write_plan",
        "  /act [notes]            implement the latest plan in a fresh context",
        "  /new                    fresh context (same engine)",
        "  /allow-all [save]       run every tool without asking (yolo)",
        "  /manual [save]          ask before edits and commands",
        "  /auto [save]            edits run, shell commands ask",
        "  /model                  rank models for this GPU;  /model <#|name|path> switches",
        "  /model auto             switch to the recommended model",
        "  /ctx [48k|64k…] [save]  show / change the context window (engine restarts)",
        "  /effort xhigh|medium|low|none   reasoning effort (xhigh/low re-prefill once)",
        "  /compact                summarize + shrink context now",
        "  /cost                   context + throughput stats",
        "  /engine status|log|restart",
        "  /reasoning              toggle reasoning display",
        "  /arcade                 open the graphical BUZZCODE ARCADE page in your browser",
        "  /factory                toggle the factory-floor side panel (F3)",
        "  /clear                  clear transcript view",
        "  /quit",
        "",
        "Permission prompt: y allow · a always allow this tool · n deny",
    ];
    let text: Vec<Line> = lines.iter().map(|l| Line::from(*l)).collect();
    f.render_widget(Paragraph::new(text).block(WBlock::default().borders(Borders::ALL).title(" Help (Esc) ")), r);
}

fn draw_engine_log(f: &mut Frame, app: &App, area: Rect) {
    let r = centered(area, 90, 80);
    f.render_widget(Clear, r);
    let lines = app.session.engine.log().snapshot();
    let h = r.height.saturating_sub(2) as usize;
    let tail: Vec<Line> = lines.iter().rev().take(h).collect::<Vec<_>>().into_iter().rev().map(|l| Line::from(l.clone())).collect();
    f.render_widget(Paragraph::new(tail).block(WBlock::default().borders(Borders::ALL).title(format!(" llama-server log · {:?} (Esc) ", app.engine_state))), r);
}

fn fmt_k(n: u32) -> String { if n >= 1000 { format!("{:.1}k", n as f64 / 1000.0) } else { n.to_string() } }
