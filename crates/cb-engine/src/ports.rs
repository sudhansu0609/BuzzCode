//! The shared Buzzcaf port ledger, from Rust.
//!
//! GUARDIAN_PLAN.md section 11 is the contract, and it is the same one
//! `dexter/backend/buzzcaf_ports.py` and `BuzzEdit/scripts/buzzcaf-ports.mjs`
//! implement for their languages. In one paragraph:
//!
//! * `engine.port` in `config.toml` is a **wish**. If something already holds
//!   it, we take the next free port — we never evict whoever is there. "Busy"
//!   is decided by a real `bind` on `127.0.0.1`, not by a connect probe, because
//!   a probe calls a TIME_WAIT socket free and then `llama-server` fails to bind
//!   anyway.
//! * Once `/health` answers we write where we landed into
//!   `%LOCALAPPDATA%\Buzzcaf\ports.json` (override `BUZZCAF_PORTS_FILE`), so
//!   Dexter, the Studio and everything else can find the engine without being
//!   told. The write is read-merge-write under `ports.json.lock` and atomic, so
//!   another app's entry is never lost to ours.
//! * The entry is removed on a clean stop. A stale entry is not a bug: readers
//!   check that the pid is alive and that the health route names the app.
//!
//! Nothing here is fatal. A ledger that cannot be written is a discovery
//! problem, never a reason the engine fails to start.

use serde_json::{Map, Value};
use std::fs;
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The name buzzcode publishes under, and the `app` its `/health` must answer.
pub const APP: &str = "buzzcode";
/// How far past the preferred port we are willing to walk before asking the OS.
pub const SPAN: u16 = 20;

const LOCK_STALE: Duration = Duration::from_secs(5);
const LOCK_RETRY: Duration = Duration::from_millis(20);
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);

// --------------------------------------------------------------------------------------------
// where the ledger lives
// --------------------------------------------------------------------------------------------

/// `%LOCALAPPDATA%\Buzzcaf\ports.json`, or whatever `BUZZCAF_PORTS_FILE` says.
pub fn ledger_path() -> PathBuf {
    resolve_ledger(
        std::env::var("BUZZCAF_PORTS_FILE").ok().as_deref(),
        std::env::var("LOCALAPPDATA").ok().as_deref(),
        std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok().as_deref(),
    )
}

/// The rule on its own, so a test can pin it without touching the process
/// environment other threads are reading (setting env vars is `unsafe` in the
/// 2024 edition for exactly that reason).
fn resolve_ledger(from_env: Option<&str>, local_appdata: Option<&str>, home: Option<&str>) -> PathBuf {
    if let Some(p) = from_env.map(str::trim).filter(|s| !s.is_empty()) {
        return PathBuf::from(p);
    }
    let base = match local_appdata.map(str::trim).filter(|s| !s.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(home.unwrap_or(".")).join(".local").join("share"),
    };
    base.join("Buzzcaf").join("ports.json")
}

// --------------------------------------------------------------------------------------------
// stepping forward
// --------------------------------------------------------------------------------------------

/// Can we bind this port right now?
///
/// A real bind. The listener is dropped immediately, which leaves a short gap
/// before `llama-server` binds it for real — unavoidable when one process picks
/// on another's behalf, and far cheaper than the alternative the plan forbids
/// (killing whoever holds the port).
pub fn bindable(port: u16, host: &str) -> bool {
    let addr: Ipv4Addr = host.parse().unwrap_or(Ipv4Addr::LOCALHOST);
    TcpListener::bind(SocketAddr::from((addr, port))).is_ok()
}

/// The preferred port if it is free, else the next free one in
/// `preferred ..= preferred + span`, else whatever the OS hands out.
pub fn pick_port(preferred: u16, span: u16, host: &str) -> u16 {
    if preferred > 0 {
        let last = preferred.saturating_add(span);
        for port in preferred..=last {
            if bindable(port, host) {
                return port;
            }
        }
    }
    // Everything in the window is taken: let the OS name one. An odd port beats
    // a startup that gives up, and `publish` tells everyone where it went.
    let addr: Ipv4Addr = host.parse().unwrap_or(Ipv4Addr::LOCALHOST);
    TcpListener::bind(SocketAddr::from((addr, 0)))
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
        .unwrap_or(preferred)
}

// --------------------------------------------------------------------------------------------
// the lock
// --------------------------------------------------------------------------------------------

/// `ports.json.lock`, created O_EXCL, retried, broken when stale.
///
/// A crashed writer must not wedge every other app on the machine, so a lock
/// older than five seconds is taken away from it. Failing to acquire is not an
/// error: we go ahead anyway, because losing one entry to a race is a smaller
/// failure than an engine that refuses to serve.
struct FileLock {
    path: PathBuf,
    held: bool,
}

impl FileLock {
    fn acquire(ledger: &Path) -> Self {
        let path = lock_path(ledger);
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let deadline = SystemTime::now() + LOCK_TIMEOUT;
        let mut held = false;
        while SystemTime::now() < deadline {
            match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    let _ = write!(f, "{}", std::process::id());
                    held = true;
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if stale(&path) {
                        let _ = fs::remove_file(&path);
                    }
                    std::thread::sleep(LOCK_RETRY);
                }
                Err(_) => break,
            }
        }
        Self { path, held }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        if self.held {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn lock_path(ledger: &Path) -> PathBuf {
    let mut s = ledger.as_os_str().to_os_string();
    s.push(".lock");
    PathBuf::from(s)
}

fn stale(lock: &Path) -> bool {
    fs::metadata(lock)
        .and_then(|m| m.modified())
        .map(|t| t.elapsed().map(|e| e > LOCK_STALE).unwrap_or(false))
        .unwrap_or(true)
}

// --------------------------------------------------------------------------------------------
// reading and writing
// --------------------------------------------------------------------------------------------

/// The whole ledger. A missing, empty or corrupt file reads as `{}` — every
/// entry in it is re-published by its owner on the next launch, so rebuilding
/// costs nothing and refusing to start costs a lot.
pub fn read_at(ledger: &Path) -> Map<String, Value> {
    let Ok(text) = fs::read_to_string(ledger) else { return Map::new() };
    // A file hand-edited in Notepad, or written by PowerShell's `-Encoding utf8`,
    // arrives with a BOM that serde_json will not look past.
    match serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

/// The whole ledger, from the default location.
pub fn entries() -> Map<String, Value> {
    read_at(&ledger_path())
}

/// One app's entry, or `None`.
pub fn entry(app: &str) -> Option<Value> {
    entry_at(&ledger_path(), app)
}

pub fn entry_at(ledger: &Path, app: &str) -> Option<Value> {
    match read_at(ledger).get(app) {
        Some(Value::Object(o)) => Some(Value::Object(o.clone())),
        _ => None,
    }
}

/// The port an app published, if it published one.
pub fn published_port(ledger: &Path, app: &str) -> Option<u16> {
    entry_at(ledger, app)?
        .get("port")?
        .as_u64()
        .and_then(|n| u16::try_from(n).ok())
}

/// temp + rename, so a reader never sees half a JSON document.
fn write_at(ledger: &Path, data: &Map<String, Value>) -> std::io::Result<()> {
    if let Some(dir) = ledger.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut tmp = ledger.as_os_str().to_os_string();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let mut body = serde_json::to_vec_pretty(data).unwrap_or_else(|_| b"{}".to_vec());
    body.push(b'\n');
    fs::write(&tmp, body)?;
    // On Windows this is MoveFileEx with MOVEFILE_REPLACE_EXISTING, so an
    // existing ledger is replaced rather than refused.
    fs::rename(&tmp, ledger)
}

// --------------------------------------------------------------------------------------------
// publish / withdraw
// --------------------------------------------------------------------------------------------

/// Build the entry the plan documents: `{port, pid, started_at, health, extra}`.
pub fn record(port: u16, health: &str, extra: Map<String, Value>) -> Value {
    let mut m = Map::new();
    m.insert("port".into(), Value::from(port));
    m.insert("pid".into(), Value::from(std::process::id()));
    m.insert(
        "started_at".into(),
        Value::from(humantime::format_rfc3339_seconds(SystemTime::now()).to_string()),
    );
    m.insert("health".into(), Value::from(health));
    m.insert("extra".into(), Value::Object(extra));
    Value::Object(m)
}

/// Record where this engine landed. Call it once `/health` answers — an entry
/// written before the socket opens is a promise nobody can keep.
pub fn publish_at(ledger: &Path, app: &str, port: u16, health: &str, extra: Map<String, Value>) {
    let _lock = FileLock::acquire(ledger);
    let mut data = read_at(ledger);
    data.insert(app.to_string(), record(port, health, extra));
    if let Err(e) = write_at(ledger, &data) {
        tracing::warn!("could not write the port ledger {}: {e}", ledger.display());
    }
}

pub fn publish(app: &str, port: u16, health: &str, extra: Map<String, Value>) {
    publish_at(&ledger_path(), app, port, health, extra);
}

/// Remove this app's entry, and only this app's entry.
pub fn withdraw_at(ledger: &Path, app: &str) {
    let _lock = FileLock::acquire(ledger);
    let mut data = read_at(ledger);
    if data.remove(app).is_some() {
        if let Err(e) = write_at(ledger, &data) {
            tracing::warn!("could not update the port ledger {}: {e}", ledger.display());
        }
    }
}

pub fn withdraw(app: &str) {
    withdraw_at(&ledger_path(), app);
}

/// `extra` for a buzzcode entry: which model, under which alias.
pub fn engine_extra(alias: &str, model: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("alias".into(), Value::from(alias));
    m.insert("model".into(), Value::from(model));
    m
}

/// The health URL for an engine on this host and port.
pub fn health_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}/health")
}

// --------------------------------------------------------------------------------------------
// tests
// --------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A ledger of our own, so a test never touches the machine's real one.
    fn temp_ledger(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("buzzcaf-ports-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("ports.json")
    }

    fn free_port() -> u16 {
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap().port()
    }

    #[test]
    fn the_env_override_wins_and_the_default_is_under_localappdata() {
        assert_eq!(
            resolve_ledger(Some(r"C:	mp\ports.json"), Some(r"C:\AppData"), None),
            PathBuf::from(r"C:	mp\ports.json"),
        );
        // Blank is not an override.
        assert_eq!(
            resolve_ledger(Some("   "), Some(r"C:\AppData"), None),
            PathBuf::from(r"C:\AppData").join("Buzzcaf").join("ports.json"),
        );
        assert_eq!(
            resolve_ledger(None, None, Some("/home/me")),
            PathBuf::from("/home/me").join(".local").join("share").join("Buzzcaf").join("ports.json"),
        );
        // And the live one agrees with the shape the other two languages use.
        assert!(ledger_path().ends_with("ports.json"));
        assert_eq!(ledger_path().parent().unwrap().file_name().unwrap(), "Buzzcaf");
    }

    #[test]
    fn a_free_port_is_taken_as_is() {
        // `free_port` closes its listener before we ask, so on a busy machine
        // somebody else can take that number in the gap - and `pick_port` then
        // steps forward, which is exactly right, while the test fails for it.
        // Seen for real in the P4 integration run with several backends coming
        // up at once (wanted 54912, got 54914). Retry with a fresh number
        // rather than pretend the gap is not there; losing the race five times
        // running is not a busy machine, it is a broken `pick_port`.
        for attempt in 0..5 {
            let p = free_port();
            let got = pick_port(p, 5, "127.0.0.1");
            if got == p {
                return;
            }
            assert!(attempt < 4, "pick_port kept stepping past a free port: {p} -> {got}");
        }
    }

    #[test]
    fn a_busy_port_is_stepped_over_and_left_alone() {
        let p = free_port();
        let squatter = TcpListener::bind((Ipv4Addr::LOCALHOST, p)).unwrap();
        let got = pick_port(p, 5, "127.0.0.1");
        assert_ne!(got, p, "we must not claim the port someone else holds");
        assert!(got > p && got <= p + 5, "expected p+1..=p+5, got {got}");
        // Still listening: stepping forward never evicts.
        assert_eq!(squatter.local_addr().unwrap().port(), p);
    }

    #[test]
    fn an_exhausted_window_falls_back_to_an_os_port() {
        let p = free_port();
        let mut held = Vec::new();
        for port in p..=p + 2 {
            if let Ok(l) = TcpListener::bind((Ipv4Addr::LOCALHOST, port)) {
                held.push(l);
            }
        }
        let got = pick_port(p, 2, "127.0.0.1");
        assert!(got > 0);
        assert!(!held.iter().any(|l| l.local_addr().unwrap().port() == got));
    }

    #[test]
    fn publishing_writes_the_documented_shape() {
        let ledger = temp_ledger("shape");
        publish_at(&ledger, APP, 8089, "http://127.0.0.1:8089/health", engine_extra("qwen3.8-27b", "m.gguf"));
        let e = entry_at(&ledger, APP).expect("published");
        assert_eq!(e["port"], 8089);
        assert_eq!(e["pid"], std::process::id());
        assert_eq!(e["health"], "http://127.0.0.1:8089/health");
        assert_eq!(e["extra"]["alias"], "qwen3.8-27b");
        assert_eq!(e["extra"]["model"], "m.gguf");
        let started = e["started_at"].as_str().unwrap();
        assert!(started.contains('T') && started.ends_with('Z'), "{started}");
    }

    #[test]
    fn publishing_merges_and_never_drops_another_app() {
        let ledger = temp_ledger("merge");
        // Somebody else got here first.
        fs::write(
            &ledger,
            r#"{"dexter":{"port":8098,"pid":1,"started_at":"t","health":"h","extra":{}}}"#,
        )
        .unwrap();
        publish_at(&ledger, APP, 8089, "http://127.0.0.1:8089/health", Map::new());
        let all = read_at(&ledger);
        assert_eq!(all.len(), 2, "{all:?}");
        assert_eq!(all["dexter"]["port"], 8098);
        assert_eq!(all[APP]["port"], 8089);
    }

    #[test]
    fn republishing_replaces_only_our_own_entry() {
        let ledger = temp_ledger("republish");
        publish_at(&ledger, "dexter", 8098, "h", Map::new());
        publish_at(&ledger, APP, 8089, "h", Map::new());
        publish_at(&ledger, APP, 8090, "h", Map::new());
        assert_eq!(published_port(&ledger, APP), Some(8090));
        assert_eq!(published_port(&ledger, "dexter"), Some(8098));
    }

    #[test]
    fn withdrawing_removes_only_our_own_entry_and_is_idempotent() {
        let ledger = temp_ledger("withdraw");
        publish_at(&ledger, "dexter", 8098, "h", Map::new());
        publish_at(&ledger, APP, 8089, "h", Map::new());
        withdraw_at(&ledger, APP);
        withdraw_at(&ledger, APP);
        withdraw_at(&ledger, "never-published");
        assert_eq!(entry_at(&ledger, APP), None);
        assert_eq!(published_port(&ledger, "dexter"), Some(8098));
    }

    #[test]
    fn a_corrupt_ledger_is_rebuilt_rather_than_fatal() {
        let ledger = temp_ledger("corrupt");
        fs::write(&ledger, b"{ not json at all").unwrap();
        publish_at(&ledger, APP, 8089, "h", Map::new());
        assert_eq!(published_port(&ledger, APP), Some(8089));
    }

    #[test]
    fn a_ledger_with_a_byte_order_mark_still_parses() {
        let ledger = temp_ledger("bom");
        fs::write(&ledger, format!("\u{feff}{}", r#"{"buzzcode":{"port":9001}}"#)).unwrap();
        assert_eq!(published_port(&ledger, APP), Some(9001));
    }

    #[test]
    fn a_missing_ledger_reads_as_nothing_published() {
        let ledger = std::env::temp_dir().join("buzzcaf-ports-absent").join("ports.json");
        let _ = fs::remove_file(&ledger);
        assert!(read_at(&ledger).is_empty());
        assert_eq!(entry_at(&ledger, APP), None);
    }

    #[test]
    fn a_stale_lock_is_broken_and_the_lock_is_released_afterwards() {
        let ledger = temp_ledger("stale-lock");
        let lock = lock_path(&ledger);
        fs::write(&lock, b"999999").unwrap();
        // Backdate it well past the five-second staleness window.
        let old = SystemTime::now() - Duration::from_secs(60);
        filetime_set(&lock, old);
        publish_at(&ledger, APP, 8089, "h", Map::new());
        assert_eq!(published_port(&ledger, APP), Some(8089));
        assert!(!lock.exists(), "the lock is released when the guard drops");
    }

    #[test]
    fn concurrent_publishers_all_survive() {
        let ledger = temp_ledger("concurrent");
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let ledger = ledger.clone();
                std::thread::spawn(move || {
                    publish_at(&ledger, &format!("app{i}"), 9000 + i as u16, "h", Map::new());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(read_at(&ledger).len(), 8, "{:?}", read_at(&ledger));
    }

    #[test]
    fn the_health_url_and_extra_are_what_the_family_expects() {
        assert_eq!(health_url("127.0.0.1", 8089), "http://127.0.0.1:8089/health");
        let e = engine_extra("a", "m.gguf");
        assert_eq!(e["alias"], "a");
        assert_eq!(e["model"], "m.gguf");
    }

    /// Backdate a file's mtime without pulling in a crate for it.
    fn filetime_set(path: &Path, when: SystemTime) {
        // `File::set_times` landed in Rust 1.75 and is all we need here.
        let f = fs::OpenOptions::new().write(true).open(path).unwrap();
        let times = fs::FileTimes::new().set_modified(when).set_accessed(when);
        f.set_times(times).unwrap();
    }
}
