use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use similar::TextDiff;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path of the file to edit.
    pub path: String,
    /// Exact text to find (must match byte-for-byte including indentation; include enough surrounding lines to be unique).
    pub old_string: String,
    /// Replacement text.
    pub new_string: String,
    /// Replace every occurrence instead of requiring a unique match.
    #[serde(default)]
    pub replace_all: bool,
}

pub struct EditFile;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "edit_file",
    "Replace an exact string in a file. old_string must appear exactly once (or set replace_all). Whitespace-sensitive. After the edit you get back the modified region with line numbers.",
    PermissionClass::WriteFs,
));

#[derive(Debug)]
struct Planned {
    new_text: String,
    count: usize,
    first_byte: usize,
}

fn plan(text: &str, a: &Args) -> Result<Planned, String> {
    if a.old_string.is_empty() { return Err("old_string is empty; use write_file to create content".into()); }
    if a.old_string == a.new_string { return Err("old_string and new_string are identical".into()); }
    // Try exact; then tolerate CRLF/LF mismatch; then tolerate trailing-whitespace-per-line mismatch.
    let variants: Vec<(String, &str)> = vec![
        (a.old_string.clone(), "exact"),
        (a.old_string.replace("\r\n", "\n"), "lf"),
        (a.old_string.replace('\n', "\r\n"), "crlf"),
    ];
    for (needle, _) in &variants {
        let count = text.matches(needle.as_str()).count();
        if count == 0 { continue; }
        if count > 1 && !a.replace_all {
            let lines: Vec<u32> = text.match_indices(needle.as_str()).map(|(i, _)| fsutil::line_number_of_byte(text, i)).collect();
            return Err(format!("old_string matches {count} times (lines {:?}); include more context to make it unique or set replace_all=true", lines));
        }
        let new_string = if needle.contains("\r\n") { a.new_string.replace("\r\n", "\n").replace('\n', "\r\n") } else { a.new_string.clone() };
        let first_byte = text.find(needle.as_str()).unwrap();
        let new_text = if a.replace_all { text.replace(needle.as_str(), &new_string) } else { text.replacen(needle.as_str(), &new_string, 1) };
        return Ok(Planned { new_text, count, first_byte });
    }
    // Whitespace-insensitive fuzzy hint.
    let hint = nearest_match_hint(text, &a.old_string);
    Err(format!("old_string not found in file.{hint}"))
}

/// Find the text region most similar to `needle` (by normalized-line comparison) to help the model fix its edit.
fn nearest_match_hint(text: &str, needle: &str) -> String {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let needle_lines: Vec<String> = needle.lines().map(norm).filter(|l| !l.is_empty()).collect();
    if needle_lines.is_empty() { return String::new(); }
    let lines: Vec<&str> = text.lines().collect();
    let n = needle_lines.len();
    let mut best = (0usize, 0usize);
    for start in 0..lines.len() {
        let mut score = 0;
        for (k, nl) in needle_lines.iter().enumerate() {
            if let Some(l) = lines.get(start + k) { if norm(l) == *nl { score += 1; } }
        }
        if score > best.0 { best = (score, start); }
    }
    if best.0 == 0 { return " No similar region found; re-read the file.".into(); }
    let s = best.1;
    let e = (s + n + 2).min(lines.len());
    let region: Vec<String> = (s.saturating_sub(1)..e).map(|i| format!("{:>4}\t{}", i + 1, lines[i])).collect();
    format!(" Closest match ({}/{} lines equal ignoring whitespace) near line {}:\n{}", best.0, n, s + 1, region.join("\n"))
}

fn region_with_numbers(text: &str, from_line: u32, to_line: u32, pad: u32) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let a = from_line.saturating_sub(pad).max(1) as usize;
    let b = (to_line + pad).min(lines.len() as u32) as usize;
    (a..=b).filter_map(|i| lines.get(i - 1).map(|l| format!("{:>4}\t{}", i, l))).collect::<Vec<_>>().join("\n")
}

#[async_trait::async_trait]
impl Tool for EditFile {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn preview(&self, args: &Value, cx: &ToolCtx) -> Result<Option<String>, ToolError> {
        let a: Args = parse_args(args)?;
        let path = fsutil::resolve(cx, &a.path);
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| ToolError::Failed(format!("cannot read {}: {e}", fsutil::display(cx, &path))))?;
        match plan(&text, &a) {
            Ok(p) => {
                let name = fsutil::display(cx, &path);
                let d = TextDiff::from_lines(&text, &p.new_text);
                Ok(Some(d.unified_diff().context_radius(3).header(&format!("a/{name}"), &format!("b/{name}")).to_string()))
            }
            Err(e) => Ok(Some(format!("(edit will fail: {e})"))),
        }
    }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = fsutil::resolve(cx, &a.path);
        fsutil::check_write_allowed(cx, &path)?;
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| ToolError::Failed(format!("cannot read {}: {e}", fsutil::display(cx, &path))))?;
        let p = match plan(&text, &a) { Ok(p) => p, Err(e) => return Ok(ToolOutput::error(e)) };
        tokio::fs::write(&path, &p.new_text).await?;
        let start_line = fsutil::line_number_of_byte(&p.new_text, p.first_byte);
        let new_lines = a.new_string.lines().count().max(1) as u32;
        let end_line = start_line + new_lines - 1;
        let region = region_with_numbers(&p.new_text, start_line, end_line, 4);
        let name = fsutil::display(cx, &path);
        let msg = if a.replace_all && p.count > 1 {
            format!("Replaced {} occurrences in {name}. First region now:\n{region}", p.count)
        } else {
            format!("Edited {name} lines {start_line}-{end_line}. Region now:\n{region}")
        };
        let mut out = ToolOutput::text(msg);
        out.edits.push(EditRecord { path, line_start: start_line, line_end: end_line, created: false });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unique_replace_and_ambiguity() {
        let t = "a\nfoo\nb\nfoo\n";
        let a = Args { path: "x".into(), old_string: "foo".into(), new_string: "bar".into(), replace_all: false };
        assert!(plan(t, &a).unwrap_err().contains("matches 2 times"));
        let a2 = Args { replace_all: true, ..a };
        assert_eq!(plan(t, &a2).unwrap().new_text, "a\nbar\nb\nbar\n");
        let a3 = Args { path: "x".into(), old_string: "b\nfoo".into(), new_string: "b\nbaz".into(), replace_all: false };
        assert_eq!(plan(t, &a3).unwrap().new_text, "a\nfoo\nb\nbaz\n");
    }
    #[test]
    fn crlf_tolerant() {
        let t = "line1\r\nline2\r\n";
        let a = Args { path: "x".into(), old_string: "line1\nline2".into(), new_string: "x\ny".into(), replace_all: false };
        assert_eq!(plan(t, &a).unwrap().new_text, "x\r\ny\r\n");
    }
}
