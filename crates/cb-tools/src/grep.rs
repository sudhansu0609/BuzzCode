//! grep via ripgrep (`rg --json`), grouped by file, capped.

use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Regular expression (ripgrep/Rust syntax).
    pub pattern: String,
    /// File or directory to search (default: project root).
    #[serde(default)]
    pub path: Option<String>,
    /// Restrict to files matching this glob, e.g. "*.rs".
    #[serde(default)]
    pub glob: Option<String>,
    /// Lines of context around each match (default 0).
    #[serde(default)]
    pub context: Option<u32>,
    /// Maximum matches to return (default 200).
    #[serde(default)]
    pub max_results: Option<u32>,
    /// Case-insensitive search.
    #[serde(default)]
    pub ignore_case: bool,
}

pub struct Grep;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "grep",
    "Search file contents with a regex (ripgrep). Respects .gitignore. Output: path:line: text, grouped by file.",
    PermissionClass::ReadOnly,
));

fn rg_path() -> String {
    std::env::var("BUZZCODE_RG").unwrap_or_else(|_| "rg".into())
}

#[async_trait::async_trait]
impl Tool for Grep {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let target = a.path.as_deref().map(|p| fsutil::resolve(cx, p)).unwrap_or(cx.cwd.clone());
        let max = a.max_results.unwrap_or(200).clamp(1, 2000) as usize;
        let mut cmd = tokio::process::Command::new(rg_path());
        cmd.arg("--json").arg("--no-messages").arg("--max-columns").arg("400").arg("--max-columns-preview")
            .arg("--max-count").arg("50")
            .arg("-e").arg(&a.pattern);
        if a.ignore_case { cmd.arg("-i"); }
        if let Some(g) = &a.glob { cmd.arg("--glob").arg(g); }
        if let Some(c) = a.context { if c > 0 { cmd.arg("-C").arg(c.min(5).to_string()); } }
        cmd.arg("--").arg(&target);
        cmd.current_dir(&cx.cwd).stdin(std::process::Stdio::null()).kill_on_drop(true);
        let out = tokio::time::timeout(std::time::Duration::from_secs(60), cmd.output()).await
            .map_err(|_| ToolError::Timeout(60))?
            .map_err(|e| ToolError::Failed(format!("ripgrep failed to start ({e}); is `rg` installed?")))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        // Group by file preserving first-seen order.
        let mut files: Vec<String> = Vec::new();
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut n_matches = 0usize;
        let mut truncated = false;
        for line in stdout.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
            let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
            if ty != "match" && ty != "context" { continue; }
            let d = &v["data"];
            let path = d["path"]["text"].as_str().unwrap_or("?").to_string();
            let rel = fsutil::display(cx, std::path::Path::new(&path));
            let ln = d["line_number"].as_u64().unwrap_or(0);
            let text = d["lines"]["text"].as_str().unwrap_or("").trim_end_matches(['\n', '\r']);
            if ty == "match" {
                n_matches += 1;
                if n_matches > max { truncated = true; break; }
            }
            if !groups.contains_key(&rel) { files.push(rel.clone()); }
            let sep = if ty == "match" { ':' } else { '-' };
            groups.entry(rel).or_default().push(format!("{ln}{sep} {}", text.chars().take(300).collect::<String>()));
        }
        if n_matches == 0 {
            let err = String::from_utf8_lossy(&out.stderr);
            if !err.trim().is_empty() && !out.status.success() && out.status.code() != Some(1) {
                return Ok(ToolOutput::error(format!("rg error: {}", err.trim())));
            }
            return Ok(ToolOutput::text(format!("No matches for {:?}{}", a.pattern, a.glob.as_ref().map(|g| format!(" in {g}")).unwrap_or_default())));
        }
        let mut text = String::new();
        for f in files {
            text.push_str(&f); text.push('\n');
            for l in &groups[&f] { text.push_str("  "); text.push_str(l); text.push('\n'); }
        }
        if truncated { text.push_str(&format!("… more than {max} matches; narrow the pattern, add a glob, or raise max_results.\n")); }
        Ok(ToolOutput::text(text))
    }
}
