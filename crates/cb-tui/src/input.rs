//! Minimal multi-line input editor (no external widget crate → no ratatui version coupling).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph};
use unicode_width::UnicodeWidthChar;

#[derive(Debug, Clone)]
pub struct InputBox {
    lines: Vec<Vec<char>>,
    row: usize,
    col: usize,
    pub placeholder: String,
    scroll: usize,
    col_scroll: usize,
}

impl Default for InputBox {
    fn default() -> Self { Self { lines: vec![Vec::new()], row: 0, col: 0, placeholder: String::new(), scroll: 0, col_scroll: 0 } }
}

impl InputBox {
    pub fn with_placeholder(p: &str) -> Self { Self { placeholder: p.into(), ..Default::default() } }

    pub fn text(&self) -> String { self.lines.iter().map(|l| l.iter().collect::<String>()).collect::<Vec<_>>().join("\n") }
    pub fn is_empty(&self) -> bool { self.lines.len() == 1 && self.lines[0].is_empty() }
    pub fn line_count(&self) -> usize { self.lines.len() }

    pub fn set_text(&mut self, s: &str) {
        self.lines = s.split('\n').map(|l| l.chars().collect()).collect();
        if self.lines.is_empty() { self.lines.push(Vec::new()); }
        self.row = self.lines.len() - 1;
        self.col = self.lines[self.row].len();
        self.col_scroll = 0;
    }

    pub fn clear(&mut self) { self.lines = vec![Vec::new()]; self.row = 0; self.col = 0; self.scroll = 0; self.col_scroll = 0; }

    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() { if c == '\n' { self.newline(); } else if c != '\r' { self.insert_char(c); } }
    }

    pub fn insert_char(&mut self, c: char) { self.lines[self.row].insert(self.col, c); self.col += 1; }

    pub fn newline(&mut self) {
        let rest: Vec<char> = self.lines[self.row].split_off(self.col);
        self.lines.insert(self.row + 1, rest);
        self.row += 1; self.col = 0; self.col_scroll = 0;
    }

    fn backspace(&mut self) {
        if self.col > 0 { self.col -= 1; self.lines[self.row].remove(self.col); }
        else if self.row > 0 {
            let cur = self.lines.remove(self.row);
            self.row -= 1;
            self.col = self.lines[self.row].len();
            self.lines[self.row].extend(cur);
        }
    }

    fn delete(&mut self) {
        if self.col < self.lines[self.row].len() { self.lines[self.row].remove(self.col); }
        else if self.row + 1 < self.lines.len() { let next = self.lines.remove(self.row + 1); self.lines[self.row].extend(next); }
    }

    fn delete_word_back(&mut self) {
        let line = &mut self.lines[self.row];
        let mut i = self.col;
        while i > 0 && line[i - 1].is_whitespace() { i -= 1; }
        while i > 0 && !line[i - 1].is_whitespace() { i -= 1; }
        line.drain(i..self.col);
        self.col = i;
    }

    /// Returns true if the key was consumed.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char(c) if !ctrl => { self.insert_char(c); }
            KeyCode::Char('w') if ctrl => self.delete_word_back(),
            KeyCode::Char('a') if ctrl => self.col = 0,
            KeyCode::Char('e') if ctrl => self.col = self.lines[self.row].len(),
            KeyCode::Char('u') if ctrl => { self.lines[self.row].drain(..self.col); self.col = 0; }
            KeyCode::Char('k') if ctrl => { self.lines[self.row].truncate(self.col); }
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete(),
            KeyCode::Left => { if self.col > 0 { self.col -= 1; } else if self.row > 0 { self.row -= 1; self.col = self.lines[self.row].len(); } }
            KeyCode::Right => { if self.col < self.lines[self.row].len() { self.col += 1; } else if self.row + 1 < self.lines.len() { self.row += 1; self.col = 0; } }
            KeyCode::Up => { if self.row > 0 { self.row -= 1; self.col = self.col.min(self.lines[self.row].len()); } else { return false; } }
            KeyCode::Down => { if self.row + 1 < self.lines.len() { self.row += 1; self.col = self.col.min(self.lines[self.row].len()); } else { return false; } }
            KeyCode::Home => self.col = 0,
            KeyCode::End => self.col = self.lines[self.row].len(),
            KeyCode::Tab => { self.insert_str("    "); }
            _ => return false,
        }
        true
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect, block: Block<'_>) {
        let inner = block.inner(area);
        f.render_widget(block, area);
        let h = inner.height.max(1) as usize;
        let w = inner.width.max(1) as usize;
        if self.row < self.scroll { self.scroll = self.row; }
        if self.row >= self.scroll + h { self.scroll = self.row + 1 - h; }

        let cursor_x: usize = self.lines[self.row][..self.col].iter().map(|c| c.width().unwrap_or(1)).sum();
        if cursor_x < self.col_scroll {
            self.col_scroll = cursor_x;
        } else if cursor_x >= self.col_scroll + w {
            self.col_scroll = cursor_x + 1 - w;
        }

        let lines: Vec<Line> = if self.is_empty() {
            vec![Line::from(Span::styled(self.placeholder.clone(), Style::default().fg(Color::DarkGray)))]
        } else {
            self.lines.iter().skip(self.scroll).take(h).enumerate().map(|(idx, l)| {
                let row_idx = self.scroll + idx;
                let hscroll = if row_idx == self.row { self.col_scroll } else { 0 };
                let mut cur_w = 0;
                let mut s = String::new();
                for c in l {
                    let cw = c.width().unwrap_or(1);
                    if cur_w + cw > hscroll && cur_w < hscroll + w {
                        s.push(*c);
                    }
                    cur_w += cw;
                }
                Line::from(s)
            }).collect()
        };
        f.render_widget(Paragraph::new(lines), inner);

        // cursor
        let cx = inner.x + (cursor_x.saturating_sub(self.col_scroll) as u16).min(inner.width.saturating_sub(1));
        let cy = inner.y + (self.row - self.scroll) as u16;
        f.set_cursor_position((cx, cy));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiline_and_newline() {
        let mut input = InputBox::default();
        input.insert_str("line 1");
        assert_eq!(input.line_count(), 1);
        input.newline();
        input.insert_str("line 2");
        assert_eq!(input.line_count(), 2);
        assert_eq!(input.text(), "line 1\nline 2");
    }

    #[test]
    fn test_horizontal_scroll() {
        let mut input = InputBox::default();
        input.insert_str(&"a".repeat(100));
        assert_eq!(input.col, 100);
    }
}
