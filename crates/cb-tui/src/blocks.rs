//! Transcript blocks and their rendering into wrapped lines (cached per width).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone)]
pub enum Block {
    User(String),
    Assistant { content: String, reasoning: String, streaming: bool },
    Tool { name: String, args: String, output: String, is_error: bool, done: bool, collapsed: bool, truncated: bool },
    Verify { command: String, ok: bool, output: String, collapsed: bool },
    Note(String),
    Warn(String),
    Error(String),
}

pub struct Rendered {
    pub width: u16,
    pub lines: Vec<Line<'static>>,
}

pub struct TranscriptBlock {
    pub block: Block,
    cache: Option<Rendered>,
    /// Incremental state for a streaming assistant block (re-wraps only the unfinished tail).
    stream: Option<StreamCache>,
    /// Set by `invalidate()`; streaming blocks use it to refresh the tail instead of a full re-render.
    dirty: bool,
}

struct StreamCache {
    width: u16,
    show_reasoning: bool,
    /// bytes of `reasoning` already folded into `reasoning_lines` (always at a '\n' boundary)
    reasoning_done: usize,
    reasoning_lines: Vec<Line<'static>>,
    content_done: usize,
    content_lines: Vec<Line<'static>>,
    /// markdown fence state at `content_done`
    in_code: bool,
    /// assembled output (header + stable + tail)
    out: Vec<Line<'static>>,
}

impl TranscriptBlock {
    pub fn new(block: Block) -> Self { Self { block, cache: None, stream: None, dirty: true } }
    pub fn invalidate(&mut self) { self.cache = None; self.dirty = true; }

    pub fn lines(&mut self, width: u16, show_reasoning: bool) -> &[Line<'static>] {
        if let Block::Assistant { streaming: true, .. } = &self.block {
            return self.stream_lines(width, show_reasoning);
        }
        self.stream = None;
        if self.cache.as_ref().map(|c| c.width != width).unwrap_or(true) || self.dirty {
            let lines = render(&self.block, width, show_reasoning);
            self.cache = Some(Rendered { width, lines });
            self.dirty = false;
        }
        &self.cache.as_ref().unwrap().lines
    }

    fn stream_lines(&mut self, width: u16, show_reasoning: bool) -> &[Line<'static>] {
        let Block::Assistant { content, reasoning, .. } = &self.block else { unreachable!() };
        let reset = self.stream.as_ref().map(|s| s.width != width || s.show_reasoning != show_reasoning).unwrap_or(true);
        if reset {
            self.stream = Some(StreamCache { width, show_reasoning, reasoning_done: 0, reasoning_lines: Vec::new(), content_done: 0, content_lines: Vec::new(), in_code: false, out: Vec::new() });
            self.dirty = true;
        }
        if !self.dirty { return &self.stream.as_ref().unwrap().out; }
        let s = self.stream.as_mut().unwrap();
        let w = width as usize;
        let rstyle = Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC);

        // Fold completed reasoning paragraphs.
        if show_reasoning {
            if let Some(nl) = reasoning.rfind('\n') {
                let stable_end = nl + 1;
                if stable_end > s.reasoning_done {
                    s.reasoning_lines.extend(wrap(&reasoning[s.reasoning_done..nl], w, rstyle, "  ┆ "));
                    s.reasoning_done = stable_end;
                }
            }
        }
        // Fold completed content lines (markdown fence state carried in `in_code`).
        if let Some(nl) = content.rfind('\n') {
            let stable_end = nl + 1;
            if stable_end > s.content_done {
                let (lines, in_code) = render_markdownish_stateful(&content[s.content_done..nl], w, s.in_code);
                s.content_lines.extend(lines);
                s.in_code = in_code;
                s.content_done = stable_end;
            }
        }
        // Assemble: header + stable + tails (tails are tiny, so re-wrapping them is cheap).
        let mut out: Vec<Line<'static>> = Vec::with_capacity(s.reasoning_lines.len() + s.content_lines.len() + 8);
        if show_reasoning && !reasoning.is_empty() {
            out.push(Line::from(Span::styled("  ┆ thinking", rstyle)));
            out.extend(s.reasoning_lines.iter().cloned());
            let tail = &reasoning[s.reasoning_done..];
            if !tail.is_empty() { out.extend(wrap(tail, w, rstyle, "  ┆ ")); }
        }
        out.extend(s.content_lines.iter().cloned());
        let tail = &content[s.content_done..];
        if !tail.is_empty() { out.extend(render_markdownish_stateful(tail, w, s.in_code).0); }
        if content.is_empty() && reasoning.is_empty() { out.push(Line::from(Span::styled("  …", Style::default().fg(Color::DarkGray)))); }
        s.out = out;
        self.dirty = false;
        &self.stream.as_ref().unwrap().out
    }
}

fn wrap<'a>(text: &str, width: usize, style: Style, prefix: &str) -> Vec<Line<'a>> {
    let w = width.saturating_sub(prefix.width()).max(8);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        if raw.is_empty() { out.push(Line::from(Span::styled(prefix.to_string(), style))); continue; }
        for piece in textwrap::wrap(raw, textwrap::Options::new(w).break_words(true)) {
            out.push(Line::from(vec![Span::styled(prefix.to_string(), style.add_modifier(Modifier::DIM)), Span::styled(piece.into_owned(), style)]));
        }
    }
    out
}

pub fn render(block: &Block, width: u16, show_reasoning: bool) -> Vec<Line<'static>> {
    let w = width as usize;
    match block {
        Block::User(t) => {
            let mut v = vec![Line::from(Span::styled("› you", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)))];
            v.extend(wrap(t, w, Style::default().fg(Color::White), "  "));
            v.push(Line::from(""));
            v
        }
        Block::Assistant { content, reasoning, streaming } => {
            let mut v = Vec::new();
            if show_reasoning && !reasoning.is_empty() {
                v.push(Line::from(Span::styled("  ┆ thinking", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))));
                let r: String = if reasoning.len() > 4000 && !*streaming { format!("…{}", &reasoning[reasoning.len() - 4000..]) } else { reasoning.clone() };
                v.extend(wrap(&r, w, Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC), "  ┆ "));
            }
            if !content.is_empty() {
                v.extend(render_markdownish(content, w));
            } else if *streaming && reasoning.is_empty() {
                v.push(Line::from(Span::styled("  …", Style::default().fg(Color::DarkGray))));
            }
            if !*streaming { v.push(Line::from("")); }
            v
        }
        Block::Tool { name, args, output, is_error, done, collapsed, truncated } => {
            let head_style = Style::default().fg(if *is_error { Color::Red } else { Color::Yellow });
            let status = if !*done { "⟳" } else if *is_error { "✗" } else { "✓" };
            let summary = summarize_output(output);
            let mut v = vec![Line::from(vec![
                Span::styled(format!("  {status} "), head_style),
                Span::styled(name.clone(), head_style.add_modifier(Modifier::BOLD)),
                Span::styled(format!(" {}", truncate_str(args, w.saturating_sub(name.len() + 8))), Style::default().fg(Color::DarkGray)),
            ])];
            if *done {
                if *collapsed {
                    if !summary.is_empty() { v.push(Line::from(Span::styled(format!("      {summary}{}", if *truncated { " (truncated)" } else { "" }), Style::default().fg(Color::DarkGray)))); }
                } else {
                    v.extend(wrap(output, w, Style::default().fg(if *is_error { Color::LightRed } else { Color::Gray }), "      "));
                }
            }
            v
        }
        Block::Verify { command, ok, output, collapsed } => {
            let st = Style::default().fg(if *ok { Color::Green } else { Color::Red });
            let mut v = vec![Line::from(vec![Span::styled(format!("  {} verify ", if *ok { "✓" } else { "✗" }), st), Span::styled(command.clone(), Style::default().fg(Color::DarkGray))])];
            if !*collapsed || !*ok {
                let tail: Vec<&str> = output.lines().rev().take(if *collapsed { 12 } else { 200 }).collect::<Vec<_>>().into_iter().rev().collect();
                v.extend(wrap(&tail.join("\n"), w, Style::default().fg(Color::Gray), "      "));
            }
            v
        }
        Block::Note(t) => wrap(t, w, Style::default().fg(Color::DarkGray), "  "),
        Block::Warn(t) => wrap(t, w, Style::default().fg(Color::Yellow), "  ! "),
        Block::Error(t) => wrap(t, w, Style::default().fg(Color::Red), "  ✗ "),
    }
}

fn summarize_output(o: &str) -> String {
    let first = o.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    let n = o.lines().count();
    let mut s: String = first.chars().take(100).collect();
    if n > 1 { s.push_str(&format!("  (+{} lines)", n - 1)); }
    s
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.width() <= max { s.to_string() } else { let mut t: String = s.chars().take(max.saturating_sub(1)).collect(); t.push('…'); t }
}

/// Light markdown: headings, bullets, inline code, fenced code blocks.
fn render_markdownish(text: &str, w: usize) -> Vec<Line<'static>> { render_markdownish_stateful(text, w, false).0 }

/// Same, but starting from (and returning) the fence state so streaming can render incrementally.
fn render_markdownish_stateful(text: &str, w: usize, mut in_code: bool) -> (Vec<Line<'static>>, bool) {
    let mut out = Vec::new();
    for raw in text.lines() {
        if raw.trim_start().starts_with("```") { in_code = !in_code; out.push(Line::from(Span::styled(format!("  {}", raw.trim()), Style::default().fg(Color::DarkGray)))); continue; }
        if in_code {
            out.push(Line::from(Span::styled(format!("  {raw}"), Style::default().fg(Color::LightGreen))));
            continue;
        }
        let (style, prefix) = if raw.starts_with('#') { (Style::default().fg(Color::White).add_modifier(Modifier::BOLD), "  ") }
            else if raw.trim_start().starts_with("- ") || raw.trim_start().starts_with("* ") { (Style::default().fg(Color::White), "  ") }
            else { (Style::default().fg(Color::White), "  ") };
        for piece in textwrap::wrap(raw, textwrap::Options::new(w.saturating_sub(2).max(8)).break_words(true)) {
            out.push(inline_code_line(&piece, style, prefix));
        }
        if raw.is_empty() { out.push(Line::from("")); }
    }
    (out, in_code)
}

fn inline_code_line(s: &str, base: Style, prefix: &str) -> Line<'static> {
    let mut spans = vec![Span::raw(prefix.to_string())];
    let mut cur = String::new();
    let mut code = false;
    for c in s.chars() {
        if c == '`' {
            if !cur.is_empty() { spans.push(Span::styled(std::mem::take(&mut cur), if code { Style::default().fg(Color::LightCyan) } else { base })); }
            code = !code;
        } else { cur.push(c); }
    }
    if !cur.is_empty() { spans.push(Span::styled(cur, if code { Style::default().fg(Color::LightCyan) } else { base })); }
    Line::from(spans)
}
