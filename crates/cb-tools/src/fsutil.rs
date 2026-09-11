use cb_tool_api::{PermissionMode, ToolCtx, ToolError};
use std::path::{Path, PathBuf};

/// Resolve a user/model-supplied path against cwd; normalize separators.
pub fn resolve(cx: &ToolCtx, p: &str) -> PathBuf {
    let p = p.trim().trim_matches('"').replace('/', std::path::MAIN_SEPARATOR_STR);
    let pb = PathBuf::from(&p);
    if pb.is_absolute() { pb } else { cx.cwd.join(pb) }
}

/// Writes are confined to the project dir unless in yolo mode.
pub fn check_write_allowed(cx: &ToolCtx, path: &Path) -> Result<(), ToolError> {
    if cx.permission_mode == PermissionMode::Yolo { return Ok(()); }
    let root = std::path::absolute(&cx.project_dir).unwrap_or(cx.project_dir.clone());
    let target = std::path::absolute(path).unwrap_or(path.to_path_buf());
    let norm = |p: &Path| p.to_string_lossy().to_ascii_lowercase().replace('/', "\\");
    if !norm(&target).starts_with(&norm(&root)) {
        return Err(ToolError::Denied(format!("refusing to write outside the project directory ({})", root.display())));
    }
    // Never touch our own state dir or VCS internals.
    let rel = norm(&target)[norm(&root).len()..].to_string();
    if rel.starts_with("\\.git\\") || rel.starts_with("\\.buzzcode\\") {
        return Err(ToolError::Denied("refusing to write inside .git or .buzzcode".into()));
    }
    Ok(())
}

pub fn display(cx: &ToolCtx, p: &Path) -> String {
    let s = p.strip_prefix(&cx.project_dir).unwrap_or(p).to_string_lossy().replace('\\', "/");
    if s.is_empty() { ".".into() } else { s }
}

pub fn is_probably_binary(bytes: &[u8]) -> bool {
    let sample = &bytes[..bytes.len().min(8000)];
    sample.contains(&0) || sample.iter().filter(|b| **b < 9 || (**b > 13 && **b < 32)).count() * 20 > sample.len().max(1)
}

/// Detect the dominant line ending of a file so edits preserve it.
pub fn line_ending(s: &str) -> &'static str {
    let crlf = s.matches("\r\n").count();
    let lf = s.matches('\n').count().saturating_sub(crlf);
    if crlf > lf { "\r\n" } else { "\n" }
}

pub fn line_number_of_byte(s: &str, byte: usize) -> u32 {
    s[..byte.min(s.len())].bytes().filter(|b| *b == b'\n').count() as u32 + 1
}
