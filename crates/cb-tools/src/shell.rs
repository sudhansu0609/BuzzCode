//! Shell tool (PowerShell on Windows). Kills the whole process tree on timeout/cancel.

use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Command to run (PowerShell syntax on Windows). Use ';' to chain commands.
    pub command: String,
    /// Timeout in seconds (default 120, max 900).
    #[serde(default)]
    pub timeout_s: Option<u64>,
    /// Working directory (default: project root).
    #[serde(default)]
    pub cwd: Option<String>,
}

pub struct Shell;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "shell",
    "Run a shell command and return its merged stdout/stderr and exit code. Non-interactive: never run commands that wait for input. Long output is truncated (head/tail).",
    PermissionClass::Exec,
));

pub fn shell_program() -> (&'static str, Vec<&'static str>) {
    if cfg!(windows) {
        if which("pwsh").is_some() { ("pwsh", vec!["-NoProfile", "-NonInteractive", "-Command"]) }
        else { ("powershell", vec!["-NoProfile", "-NonInteractive", "-Command"]) }
    } else { ("sh", vec!["-c"]) }
}

fn which(name: &str) -> Option<std::path::PathBuf> {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join(&exe)).find(|p| p.is_file()))
}

#[async_trait::async_trait]
impl Tool for Shell {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn preview(&self, args: &Value, _cx: &ToolCtx) -> Result<Option<String>, ToolError> {
        let a: Args = parse_args(args)?;
        Ok(Some(format!("$ {}", a.command)))
    }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let cwd = a.cwd.as_deref().map(|p| fsutil::resolve(cx, p)).unwrap_or(cx.cwd.clone());
        let timeout = Duration::from_secs(a.timeout_s.unwrap_or(120).clamp(1, 900));
        let (prog, pre) = shell_program();
        let mut cmd = tokio::process::Command::new(prog);
        cmd.args(&pre).arg(&a.command).current_dir(&cwd)
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .env("TERM", "dumb").env("NO_COLOR", "1").env("CI", "1")
            .kill_on_drop(true);
        #[cfg(windows)]
        { cmd.creation_flags(0x0800_0000 /* CREATE_NO_WINDOW */ | 0x0000_0200 /* CREATE_NEW_PROCESS_GROUP */); }
        let t0 = std::time::Instant::now();
        let mut child = cmd.spawn().map_err(|e| ToolError::Failed(format!("spawn {prog}: {e}")))?;
        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let cap = cx.budget.max_bytes * 8; // keep more than we show; truncation picks head/tail
        let read_all = async {
            let mut so = Vec::new(); let mut se = Vec::new();
            let (r1, r2) = tokio::join!(read_capped(&mut stdout, &mut so, cap), read_capped(&mut stderr, &mut se, cap));
            let _ = (r1, r2);
            (so, se)
        };
        let pid = child.id();
        let result = tokio::select! {
            _ = cx.cancel.cancelled() => { kill_tree(&mut child, pid).await; return Err(ToolError::Cancelled); }
            r = tokio::time::timeout(timeout, async { let (so, se) = read_all.await; let st = child.wait().await; (so, se, st) }) => r,
        };
        match result {
            Ok((so, se, st)) => {
                let status = st.map_err(|e| ToolError::Failed(e.to_string()))?;
                let mut text = String::from_utf8_lossy(&so).to_string();
                let err = String::from_utf8_lossy(&se);
                if !err.trim().is_empty() { if !text.is_empty() && !text.ends_with('\n') { text.push('\n'); } text.push_str(&err); }
                let code = status.code().unwrap_or(-1);
                let header = format!("[exit {code} in {:.1}s]\n", t0.elapsed().as_secs_f64());
                let mut out = ToolOutput::text(header + text.trim_end());
                out.is_error = code != 0;
                Ok(out)
            }
            Err(_) => {
                kill_tree(&mut child, pid).await;
                Ok(ToolOutput::error(format!("command timed out after {}s and was killed", timeout.as_secs())))
            }
        }
    }
}

async fn read_capped<R: tokio::io::AsyncRead + Unpin>(r: &mut R, buf: &mut Vec<u8>, cap: usize) -> std::io::Result<()> {
    let mut chunk = [0u8; 8192];
    let mut dropped = 0usize;
    loop {
        let n = r.read(&mut chunk).await?;
        if n == 0 { break; }
        if buf.len() < cap { buf.extend_from_slice(&chunk[..n]); } else { dropped += n; }
    }
    if dropped > 0 { buf.extend_from_slice(format!("\n… [{dropped} more bytes dropped]\n").as_bytes()); }
    Ok(())
}

async fn kill_tree(child: &mut tokio::process::Child, pid: Option<u32>) {
    #[cfg(windows)]
    if let Some(pid) = pid {
        let _ = tokio::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).status().await;
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}
