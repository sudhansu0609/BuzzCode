//! `~/.buzzcode/engine.json` — who is serving, on what port, with which model.
//!
//! Before this existed, `buzzcode engine serve` could only be stopped by finding
//! its console and pressing Ctrl-C. Everything else on this machine that needs
//! the GPU back — Dexter, Sentinel, the owner — had no name to call and no PID
//! to aim at. The file is that name: written when the server is healthy, removed
//! on the way out, and treated as advisory (a stale file is a normal outcome of
//! a hard kill, not an error).

use anyhow::{Context, Result};
use cb_config::Config;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The contents of `engine.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineRecord {
    /// The `buzzcode` process that is serving.
    pub pid: u32,
    /// The `llama-server` child, when it has been spawned.
    #[serde(default)]
    pub server_pid: Option<u32>,
    pub port: u16,
    /// Model file name (not the full path — that is in the log).
    pub model: String,
    /// The `--alias` the server answers to, i.e. the OpenAI `model` field.
    pub alias: String,
    /// RFC 3339, seconds resolution.
    pub started_at: String,
}

impl EngineRecord {
    pub fn new(port: u16, model: impl Into<String>, alias: impl Into<String>, server_pid: Option<u32>) -> Self {
        Self {
            pid: std::process::id(),
            server_pid,
            port,
            model: model.into(),
            alias: alias.into(),
            started_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        }
    }
}

/// `~/.buzzcode/engine.json`.
pub fn path(cfg: &Config) -> PathBuf { cfg.paths.user_dir.join("engine.json") }

/// Write the record. Written to a sibling temp file first so a reader never sees
/// half a JSON document.
pub fn write(cfg: &Config, rec: &EngineRecord) -> Result<()> {
    let p = path(cfg);
    if let Some(dir) = p.parent() { std::fs::create_dir_all(dir).ok(); }
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(rec)?).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &p).with_context(|| format!("renaming into {}", p.display()))?;
    Ok(())
}

/// The record, or `None` when there is no file (or it is unreadable garbage).
pub fn read(cfg: &Config) -> Option<EngineRecord> { read_at(&path(cfg)) }

pub fn read_at(p: &Path) -> Option<EngineRecord> {
    let text = std::fs::read_to_string(p).ok()?;
    // A file hand-edited in Notepad, or written by PowerShell's `-Encoding utf8`,
    // arrives with a BOM that serde_json will not look past.
    serde_json::from_str(text.trim_start_matches('\u{feff}')).ok()
}

/// Remove the file. Missing is success.
pub fn remove(cfg: &Config) {
    let p = path(cfg);
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(p.with_extension("json.tmp"));
}

// --------------------------------------------------------------------------------------------
// process control
// --------------------------------------------------------------------------------------------

/// Is this PID alive? Best-effort: a "no" from a broken `tasklist` reads as
/// "not running", which is the safe answer for a stale-file check.
#[cfg(windows)]
pub fn alive(pid: u32) -> bool {
    let out = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")),
        Err(_) => false,
    }
}

#[cfg(not(windows))]
pub fn alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Terminate a process and everything it spawned. `true` when the OS agreed.
///
/// `llama-server` is a grandchild in some launch paths, so `/T` is not optional;
/// `/F` is, but a model server that has stopped answering `/health` is not going
/// to shut down politely either.
#[cfg(windows)]
pub fn kill_tree(pid: u32) -> bool {
    std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(not(windows))]
pub fn kill_tree(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_round_trips_through_json() {
        let rec = EngineRecord {
            pid: 111,
            server_pid: Some(222),
            port: 8089,
            model: "Qwen3.8-27B-UD-IQ3_XXS.gguf".into(),
            alias: "qwen3.8-27b".into(),
            started_at: "2026-09-10T12:00:00Z".into(),
        };
        let text = serde_json::to_string(&rec).unwrap();
        let back: EngineRecord = serde_json::from_str(&text).unwrap();
        assert_eq!(back.pid, 111);
        assert_eq!(back.server_pid, Some(222));
        assert_eq!(back.port, 8089);
        assert_eq!(back.alias, "qwen3.8-27b");
    }

    #[test]
    fn a_file_written_before_the_server_existed_still_parses() {
        // `serve` writes the record as soon as it knows the port, and fills the
        // server PID in on the next write.
        let rec: EngineRecord = serde_json::from_str(
            r#"{"pid":1,"port":8089,"model":"m.gguf","alias":"a","started_at":"2026-09-10T00:00:00Z"}"#,
        )
        .unwrap();
        assert_eq!(rec.server_pid, None);
    }

    #[test]
    fn new_stamps_this_process_and_a_readable_time() {
        let rec = EngineRecord::new(8089, "m.gguf", "a", Some(9));
        assert_eq!(rec.pid, std::process::id());
        assert!(rec.started_at.contains('T') && rec.started_at.ends_with('Z'), "{}", rec.started_at);
    }

    #[test]
    fn a_missing_file_reads_as_nothing_running() {
        let p = std::env::temp_dir().join("buzzcode-engine-does-not-exist.json");
        let _ = std::fs::remove_file(&p);
        assert!(read_at(&p).is_none());
    }

    #[test]
    fn a_file_with_a_byte_order_mark_still_parses() {
        // PowerShell 5.1's `Set-Content -Encoding utf8` writes one, and so does
        // Notepad; serde_json on its own will not look past it.
        let p = std::env::temp_dir().join(format!("buzzcode-engine-bom-{}.json", std::process::id()));
        let body = r#"{"pid":7,"port":8089,"model":"m","alias":"a","started_at":"t"}"#;
        std::fs::write(&p, format!("\u{feff}{body}")).unwrap();
        assert_eq!(read_at(&p).map(|r| r.pid), Some(7));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn garbage_reads_as_nothing_running_rather_than_an_error() {
        let p = std::env::temp_dir().join(format!("buzzcode-engine-garbage-{}.json", std::process::id()));
        std::fs::write(&p, b"{ not json").unwrap();
        assert!(read_at(&p).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn this_very_process_is_alive_and_pid_zero_is_not() {
        assert!(alive(std::process::id()));
        assert!(!alive(0xFFFF_FFF0));
    }
}
