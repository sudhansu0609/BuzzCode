//! Git tools (shell out to `git`; honours the user's hooks/config).

use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use std::process::Stdio;

async fn git(cx: &ToolCtx, args: &[&str]) -> Result<(bool, String), ToolError> {
    let out = tokio::process::Command::new("git").args(args).current_dir(&cx.cwd)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true)
        .output().await.map_err(|e| ToolError::Failed(format!("git: {e}")))?;
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    let e = String::from_utf8_lossy(&out.stderr);
    if !e.trim().is_empty() { if !s.is_empty() { s.push('\n'); } s.push_str(e.trim_end()); }
    Ok((out.status.success(), s))
}

// ---------------------------------------------------------------- status

#[derive(Deserialize, schemars::JsonSchema)]
pub struct NoArgs {}

pub struct GitStatus;
static STATUS_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<NoArgs>(
    "git_status", "Show branch and working-tree status (git status --short --branch).", PermissionClass::ReadOnly));

#[async_trait::async_trait]
impl Tool for GitStatus {
    fn spec(&self) -> &ToolSpec { &STATUS_SPEC }
    async fn call(&self, _args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let (ok, s) = git(cx, &["status", "--short", "--branch"]).await?;
        Ok(if ok { ToolOutput::text(if s.trim().is_empty() { "clean".into() } else { s }) } else { ToolOutput::error(s) })
    }
}

// ---------------------------------------------------------------- diff

#[derive(Deserialize, schemars::JsonSchema)]
pub struct DiffArgs {
    /// Show staged changes instead of unstaged.
    #[serde(default)]
    pub staged: bool,
    /// Limit to a path.
    #[serde(default)]
    pub path: Option<String>,
    /// Only list changed files with counts.
    #[serde(default)]
    pub stat: bool,
}

pub struct GitDiff;
static DIFF_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<DiffArgs>(
    "git_diff", "Show a diff of unstaged (default) or staged changes.", PermissionClass::ReadOnly));

#[async_trait::async_trait]
impl Tool for GitDiff {
    fn spec(&self) -> &ToolSpec { &DIFF_SPEC }
    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: DiffArgs = parse_args(&args)?;
        let mut v: Vec<&str> = vec!["diff", "--no-color"];
        if a.staged { v.push("--cached"); }
        if a.stat { v.push("--stat"); }
        let p;
        if let Some(path) = &a.path { v.push("--"); p = path.clone(); v.push(&p); }
        let (ok, s) = git(cx, &v).await?;
        Ok(if ok { ToolOutput::text(if s.trim().is_empty() { "no changes".into() } else { s }) } else { ToolOutput::error(s) })
    }
}

// ---------------------------------------------------------------- log

#[derive(Deserialize, schemars::JsonSchema)]
pub struct LogArgs {
    /// Number of commits (default 10).
    #[serde(default)]
    pub n: Option<u32>,
    /// Limit to a path.
    #[serde(default)]
    pub path: Option<String>,
}

pub struct GitLog;
static LOG_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<LogArgs>(
    "git_log", "Show recent commits (one line each).", PermissionClass::ReadOnly));

#[async_trait::async_trait]
impl Tool for GitLog {
    fn spec(&self) -> &ToolSpec { &LOG_SPEC }
    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: LogArgs = parse_args(&args)?;
        let n = format!("-n{}", a.n.unwrap_or(10).clamp(1, 200));
        let mut v: Vec<&str> = vec!["log", "--oneline", "--decorate", "--no-color", &n];
        let p;
        if let Some(path) = &a.path { v.push("--"); p = path.clone(); v.push(&p); }
        let (ok, s) = git(cx, &v).await?;
        Ok(if ok { ToolOutput::text(s) } else { ToolOutput::error(s) })
    }
}

// ---------------------------------------------------------------- commit

#[derive(Deserialize, schemars::JsonSchema)]
pub struct CommitArgs {
    /// Commit message. If omitted, the harness generates one from the staged diff.
    #[serde(default)]
    pub message: Option<String>,
    /// Stage all tracked changes first (git add -A).
    #[serde(default)]
    pub add_all: bool,
    /// Specific paths to stage before committing.
    #[serde(default)]
    pub paths: Vec<String>,
}

pub struct GitCommit;
static COMMIT_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<CommitArgs>(
    "git_commit", "Stage (optionally) and commit. Never bypasses hooks. Provide a message or leave it empty to have one generated.", PermissionClass::WriteFs));

/// Set by the core when it can generate messages (we keep the tool crate engine-free).
pub type MessageGenerator = dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>> + Send + Sync;
pub struct CommitMessageGen(pub Box<MessageGenerator>);

#[async_trait::async_trait]
impl Tool for GitCommit {
    fn spec(&self) -> &ToolSpec { &COMMIT_SPEC }

    async fn preview(&self, args: &Value, cx: &ToolCtx) -> Result<Option<String>, ToolError> {
        let a: CommitArgs = parse_args(args)?;
        let (_, status) = git(cx, &["status", "--short"]).await?;
        Ok(Some(format!("message: {}\nadd_all: {}\npaths: {:?}\n--- status ---\n{status}", a.message.as_deref().unwrap_or("(generated)"), a.add_all, a.paths)))
    }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: CommitArgs = parse_args(&args)?;
        if a.add_all { let (ok, s) = git(cx, &["add", "-A"]).await?; if !ok { return Ok(ToolOutput::error(s)); } }
        if !a.paths.is_empty() {
            let mut v: Vec<&str> = vec!["add", "--"];
            for p in &a.paths { v.push(p); }
            let (ok, s) = git(cx, &v).await?; if !ok { return Ok(ToolOutput::error(s)); }
        }
        let (_, staged) = git(cx, &["diff", "--cached", "--stat"]).await?;
        if staged.trim().is_empty() { return Ok(ToolOutput::error("nothing staged; use add_all=true or paths=[...]")); }
        let message = match a.message.filter(|m| !m.trim().is_empty()) {
            Some(m) => m,
            None => {
                let (_, diff) = git(cx, &["diff", "--cached", "--no-color"]).await?;
                let diff: String = diff.chars().take(24_000).collect();
                match cx.extensions.get::<CommitMessageGen>() {
                    Some(g) => (g.0)(diff).await.unwrap_or_else(|| "Update files".into()),
                    None => return Ok(ToolOutput::error("no message provided and no generator available; pass message")),
                }
            }
        };
        let (ok, s) = git(cx, &["commit", "-m", &message]).await?;
        if !ok { return Ok(ToolOutput::error(format!("commit failed:\n{s}"))); }
        let (_, head) = git(cx, &["log", "--oneline", "-n1"]).await?;
        Ok(ToolOutput::text(format!("Committed: {}\n{}", head.trim(), staged.trim())))
    }
}
