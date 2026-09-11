//! Head/tail truncation with spill-to-disk and a paging hint for the model.

use cb_tool_api::{OutputBudget, TruncInfo};
use std::path::Path;

pub struct Truncated {
    pub text: String,
    pub info: Option<TruncInfo>,
}

/// Apply the budget. Oversized output is written to `<spill_dir>/<stem>.txt` and replaced by
/// head + tail with a hint pointing at the spill file and the offset to continue from.
pub fn apply(text: &str, budget: &OutputBudget, spill_dir: &Path, stem: &str) -> Truncated {
    let lines: Vec<&str> = text.lines().collect();
    let too_many_lines = lines.len() > budget.head_lines + budget.tail_lines + 10;
    if text.len() <= budget.max_bytes && !too_many_lines {
        return Truncated { text: text.to_string(), info: None };
    }
    let _ = std::fs::create_dir_all(spill_dir);
    let spill = spill_dir.join(format!("{stem}.txt"));
    let _ = std::fs::write(&spill, text);

    let head_n = budget.head_lines.min(lines.len());
    let tail_n = budget.tail_lines.min(lines.len().saturating_sub(head_n));
    let mut out = String::new();
    let mut bytes = 0usize;
    let head_budget = budget.max_bytes * 3 / 4;
    let mut shown_head = 0;
    for l in &lines[..head_n] {
        if bytes + l.len() > head_budget { break; }
        out.push_str(l); out.push('\n'); bytes += l.len() + 1; shown_head += 1;
    }
    let omitted = lines.len().saturating_sub(shown_head + tail_n);
    out.push_str(&format!(
        "\n… [{omitted} lines omitted of {total}; full output saved to {path} — use read_file(path=\"{path}\", offset={off}) to continue] …\n\n",
        total = lines.len(), path = spill.to_string_lossy().replace('\\', "/"), off = shown_head + 1,
    ));
    let mut tail_lines: Vec<&str> = Vec::new();
    let mut tail_bytes = 0usize;
    for l in lines.iter().rev().take(tail_n) {
        if tail_bytes + l.len() > budget.max_bytes / 4 { break; }
        tail_lines.push(l); tail_bytes += l.len() + 1;
    }
    tail_lines.reverse();
    let shown_tail = tail_lines.len();
    for l in tail_lines { out.push_str(l); out.push('\n'); }
    Truncated { text: out, info: Some(TruncInfo { total_lines: lines.len(), shown_head, shown_tail, spill_path: spill }) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncates_and_spills() {
        let dir = std::env::temp_dir().join("buzzcode-trunc-test");
        let text: String = (0..1000).map(|i| format!("line {i}\n")).collect();
        let t = apply(&text, &OutputBudget { max_bytes: 2000, head_lines: 20, tail_lines: 5 }, &dir, "t1");
        assert!(t.info.is_some());
        assert!(t.text.contains("lines omitted"));
        assert!(t.text.starts_with("line 0\n"));
        assert!(t.text.trim_end().ends_with("line 999"));
        assert!(dir.join("t1.txt").exists());
    }
}
