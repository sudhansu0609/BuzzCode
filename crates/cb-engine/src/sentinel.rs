//! Ask Sentinel before taking the GPU.
//!
//! Sentinel is the machine-wide guardian of VRAM and RAM. buzzcode's engine is
//! the single biggest claimant on this box (27B ≈ 12 GiB), so it asks before it
//! spawns `llama-server` and gives the booking back the moment the server is
//! healthy and its real usage is visible to Sentinel's own accounting.
//!
//! Three rules shape everything below:
//!
//! * **Absent Sentinel = yesterday's behaviour.** Not installed, not running,
//!   slow, wrong ACL — every one of those must look like "go ahead". A guardian
//!   that is down may never be the reason a coding run does not start.
//! * **Nothing blocks for long.** Windows named-pipe I/O has no per-call
//!   timeout, so each request runs on a worker thread that we simply abandon
//!   after 300 ms. An abandoned thread costs one handle until the pipe closes.
//! * **A refusal is a refusal.** When Sentinel says no we print who is holding
//!   the card and exit 3. `--force` is the owner's override, not ours.
//!
//! The wire format is the same newline-delimited JSON the Sentinel tray speaks:
//! serde's externally-tagged enums, e.g. `{"Reserve":{"client":"buzzcode",…}}`
//! in and `{"Reserved":{"granted":false,…}}` back.

use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// The service's pipe. `SENTINEL_PIPE` overrides it (the dev service listens on
/// `\\.\pipe\SentinelDev`).
pub const DEFAULT_PIPE: &str = r"\\.\pipe\Sentinel";
/// The name we register under; every later command carries it.
pub const CLIENT: &str = "buzzcode";
/// Class and priority for a coding engine: batch work, mid-pack.
pub const CLASS: &str = "batch";
pub const PRIORITY: u32 = 30;
pub const LABEL: &str = "buzzcode engine";
/// Deliberately short: this sits in front of a model load.
pub const CALL_TIMEOUT: Duration = Duration::from_millis(300);

pub fn pipe_path() -> String {
    resolve_pipe(std::env::var("SENTINEL_PIPE").ok().as_deref())
}

/// The env-var rule, split out so a test can pin it without touching the
/// process environment other threads are reading.
fn resolve_pipe(from_env: Option<&str>) -> String {
    match from_env {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => DEFAULT_PIPE.to_string(),
    }
}

// --------------------------------------------------------------------------------------------
// process-wide switches
// --------------------------------------------------------------------------------------------

static FORCE: AtomicBool = AtomicBool::new(false);
/// `--force` on the command line: proceed even when Sentinel refuses.
pub fn set_force(v: bool) { FORCE.store(v, Ordering::Relaxed); }
pub fn forced() -> bool { FORCE.load(Ordering::Relaxed) }

static DISABLED: AtomicBool = AtomicBool::new(false);
/// Skip Sentinel entirely. Set by `BUZZCODE_SENTINEL=0` or by tests that must
/// not touch the live service.
pub fn set_disabled(v: bool) { DISABLED.store(v, Ordering::Relaxed); }
pub fn disabled() -> bool {
    DISABLED.load(Ordering::Relaxed)
        || matches!(std::env::var("BUZZCODE_SENTINEL").as_deref(), Ok("0") | Ok("off") | Ok("false"))
}

// --------------------------------------------------------------------------------------------
// wire shapes
// --------------------------------------------------------------------------------------------

/// One row of `Reserved.blockers`. Every field is optional: the shape arrived in
/// Sentinel G10/G11 and the build installed today answers without it at all.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Blocker {
    #[serde(default)] pub client: Option<String>,
    #[serde(default)] pub process: Option<String>,
    #[serde(default)] pub pid: Option<u64>,
    #[serde(default)] pub vram_mib: Option<u64>,
    #[serde(default)] pub ram_mib: Option<u64>,
    #[serde(default)] pub class: Option<String>,
    #[serde(default)] pub evictable: Option<bool>,
}

impl std::fmt::Display for Blocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let who = self.client.clone().or_else(|| self.process.clone()).unwrap_or_else(|| "?".into());
        write!(f, "{who}")?;
        if let Some(p) = self.pid { write!(f, " (pid {p})")?; }
        if let Some(v) = self.vram_mib { write!(f, " {v} MiB VRAM")?; }
        if let Some(r) = self.ram_mib { write!(f, " + {r} MiB RAM")?; }
        if let Some(c) = &self.class { write!(f, ", {c}")?; }
        if self.evictable == Some(true) { write!(f, ", evictable")?; }
        Ok(())
    }
}

/// The `Reserved` reply. Unknown fields are ignored and missing ones default, so
/// both the service installed today and a future G10+ one parse.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Reserved {
    #[serde(default)] pub granted: bool,
    #[serde(default)] pub free_mib: Option<u64>,
    #[serde(default)] pub reserved_mib: Option<u64>,
    #[serde(default)] pub ram_free_mib: Option<u64>,
    #[serde(default)] pub ram_reserved_mib: Option<u64>,
    #[serde(default)] pub expires_in_secs: Option<u64>,
    #[serde(default)] pub reason: Option<String>,
    /// `None` when the reply carried no `blockers` key at all (the pre-G10
    /// service), `Some([])` when a v2 service answered "nobody holds a
    /// booking". Those are different facts and the refusal text says so: a
    /// live integration run against the v2 build printed "not reported by this
    /// Sentinel build" while talking to a service that reports blockers
    /// perfectly well and had simply found none.
    #[serde(default)] pub blockers: Option<Vec<Blocker>>,
}

/// What came back from a `Reserve`.
#[derive(Debug, Clone)]
pub enum Decision {
    /// Sentinel did not answer. Proceed exactly as before Sentinel existed.
    Absent,
    Granted(Reserved),
    Refused(Reserved),
}

/// A refusal, already formatted for the terminal. `main` turns this into exit 3.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct Refused {
    pub message: String,
    pub free_mib: Option<u64>,
    pub blockers: Vec<Blocker>,
}

// --------------------------------------------------------------------------------------------
// client
// --------------------------------------------------------------------------------------------

pub struct Sentinel {
    pipe: String,
    client: String,
}

impl Default for Sentinel {
    fn default() -> Self { Self::new() }
}

impl Sentinel {
    pub fn new() -> Self { Self::with_pipe(pipe_path()) }

    pub fn with_pipe(pipe: impl Into<String>) -> Self {
        Self { pipe: pipe.into(), client: CLIENT.to_string() }
    }

    pub fn pipe(&self) -> &str { &self.pipe }
    pub fn client(&self) -> &str { &self.client }

    /// One command, one reply, on a connection of its own. Blocking.
    fn round_trip(pipe: &str, cmd: &Value) -> std::io::Result<Option<Value>> {
        let mut h = std::fs::OpenOptions::new().read(true).write(true).open(pipe)?;
        let mut line = serde_json::to_vec(cmd).unwrap_or_default();
        line.push(b'\n');
        h.write_all(&line)?;
        h.flush()?;
        read_json_line(&mut h)
    }

    /// `round_trip` with a deadline. `None` means "Sentinel is absent", which
    /// covers a missing pipe, a slow service and a malformed reply alike.
    fn request(&self, cmd: Value, timeout: Duration) -> Option<Value> {
        if disabled() { return None; }
        let (tx, rx) = mpsc::channel();
        let pipe = self.pipe.clone();
        let name = command_name(&cmd);
        std::thread::Builder::new()
            .name("sentinel-request".into())
            .spawn(move || { let _ = tx.send(Sentinel::round_trip(&pipe, &cmd)); })
            .ok()?;
        match rx.recv_timeout(timeout) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => { tracing::debug!("sentinel: {name} failed: {e}"); None }
            Err(_) => { tracing::debug!("sentinel: {name} timed out after {timeout:?}"); None }
        }
    }

    /// "This PID is buzzcode's engine, class batch, priority 30."
    ///
    /// Accepts both replies the protocol allows: the old `"Ok"` and G10's
    /// `{"Registered":{token}}`.
    pub fn register(&self, pid: u32) -> bool {
        let cmd = json!({"Register": {
            "client": self.client,
            "pid": pid,
            "class": CLASS,
            "priority": PRIORITY,
            "label": LABEL,
        }});
        match self.request(cmd, CALL_TIMEOUT) {
            Some(Value::String(s)) if s == "Ok" => true,
            Some(Value::Object(m)) if m.contains_key("Registered") => true,
            _ => false,
        }
    }

    /// Ask for VRAM (and the RAM the server will spill into) before spawning.
    ///
    /// `for_pid` names the process that will actually consume the booking. We
    /// only learn llama-server's PID after the spawn, so the pre-spawn call
    /// omits it and the booking is handed back by `Release` once `/health`
    /// answers.
    pub fn reserve(&self, mib: u64, ram_mib: u64, for_pid: Option<u32>) -> Decision {
        let mut body = serde_json::Map::new();
        body.insert("client".into(), json!(self.client));
        body.insert("mib".into(), json!(mib));
        body.insert("ram_mib".into(), json!(ram_mib));
        if let Some(p) = for_pid { body.insert("for_pid".into(), json!(p)); }
        let reply = self.request(json!({ "Reserve": Value::Object(body) }), CALL_TIMEOUT);
        let Some(Value::Object(mut m)) = reply else { return Decision::Absent };
        let Some(inner) = m.remove("Reserved") else { return Decision::Absent };
        let r: Reserved = match serde_json::from_value(inner) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("sentinel: unparsable Reserved: {e}");
                return Decision::Absent;
            }
        };
        if r.granted { Decision::Granted(r) } else { Decision::Refused(r) }
    }

    /// Give the booking back. Best-effort and silent.
    pub fn release(&self) {
        let _ = self.request(json!({"Release": {"client": self.client}}), CALL_TIMEOUT);
    }

    /// Stream Sentinel's events to `on_event` on a thread of its own.
    ///
    /// The connection is separate from the command connection so a reply can
    /// never be mistaken for an event. The thread is never joined: the process
    /// exiting is what ends it.
    pub fn subscribe<F>(&self, mut on_event: F) -> Subscription
    where
        F: FnMut(Value) + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        if disabled() { return Subscription { stop }; }
        let pipe = self.pipe.clone();
        let client = self.client.clone();
        let flag = stop.clone();
        let _ = std::thread::Builder::new().name("sentinel-events".into()).spawn(move || {
            let mut backoff = Duration::from_secs(1);
            while !flag.load(Ordering::Relaxed) {
                match subscribe_once(&pipe, &client, &flag, &mut on_event) {
                    Ok(()) => backoff = Duration::from_secs(1),
                    Err(e) => tracing::debug!("sentinel: event stream unavailable ({e})"),
                }
                if flag.load(Ordering::Relaxed) { return; }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        });
        Subscription { stop }
    }
}

pub struct Subscription {
    stop: Arc<AtomicBool>,
}

impl Subscription {
    /// Ask the reader thread to stop. Best-effort: it may be parked in a
    /// blocking read until the next event arrives or the process exits.
    pub fn stop(&self) { self.stop.store(true, Ordering::Relaxed); }
}

impl Drop for Subscription {
    fn drop(&mut self) { self.stop(); }
}

fn subscribe_once<F>(pipe: &str, client: &str, stop: &AtomicBool, on_event: &mut F) -> std::io::Result<()>
where
    F: FnMut(Value),
{
    let mut h = std::fs::OpenOptions::new().read(true).write(true).open(pipe)?;
    let mut line = serde_json::to_vec(&json!({"Subscribe": {"client": client}})).unwrap_or_default();
    line.push(b'\n');
    h.write_all(&line)?;
    h.flush()?;
    match read_json_line(&mut h)? {
        Some(Value::String(s)) if s == "Subscribed" => {}
        other => return Err(std::io::Error::other(format!("unexpected reply to Subscribe: {other:?}"))),
    }
    tracing::info!("sentinel: subscribed to events on {pipe}");
    while !stop.load(Ordering::Relaxed) {
        match read_json_line(&mut h)? {
            Some(ev) => on_event(ev),
            None => break,
        }
    }
    Ok(())
}

/// Read one newline-terminated JSON value. `Ok(None)` = the peer closed.
fn read_json_line<R: Read>(r: &mut R) -> std::io::Result<Option<Value>> {
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    let mut one = [0u8; 1];
    loop {
        match r.read(&mut one) {
            Ok(0) => return Ok(None),
            Ok(_) => {
                if one[0] == b'\n' { break; }
                buf.push(one[0]);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    let text = String::from_utf8_lossy(&buf).trim().to_string();
    if text.is_empty() { return Ok(None); }
    serde_json::from_str(&text).map(Some).map_err(std::io::Error::other)
}

fn command_name(cmd: &Value) -> String {
    match cmd {
        Value::String(s) => s.clone(),
        Value::Object(m) => m.keys().next().cloned().unwrap_or_else(|| "?".into()),
        other => other.to_string(),
    }
}

// --------------------------------------------------------------------------------------------
// the gate in front of a spawn
// --------------------------------------------------------------------------------------------

/// A held booking. Dropping it releases, so an early `?` on the spawn path
/// cannot leak VRAM out of Sentinel's books.
pub struct Grant {
    sentinel: Option<Sentinel>,
    released: bool,
    /// True when Sentinel never answered: nothing was booked.
    pub absent: bool,
}

impl Grant {
    fn absent() -> Self { Self { sentinel: None, released: true, absent: true } }
    fn held(s: Sentinel) -> Self { Self { sentinel: Some(s), released: false, absent: false } }

    /// Hand the booking back — call this once `/health` answers.
    pub fn release(&mut self) {
        if self.released { return; }
        self.released = true;
        if let Some(s) = &self.sentinel { s.release(); }
    }
}

impl Drop for Grant {
    fn drop(&mut self) { self.release(); }
}

/// Register + Reserve before spawning `llama-server`.
///
/// `Ok` means go (granted, absent, or `--force`d past a refusal). `Err` carries
/// the message the caller prints before exiting 3.
pub fn gate_engine_start(vram_mib: u64, ram_mib: u64) -> Result<Grant, Refused> {
    if disabled() { return Ok(Grant::absent()); }
    gate_with(Sentinel::new(), vram_mib, ram_mib)
}

/// `gate_engine_start` against a given client — the seam the tests drive.
pub(crate) fn gate_with(s: Sentinel, vram_mib: u64, ram_mib: u64) -> Result<Grant, Refused> {
    // A failed Register is not fatal: it usually just means Sentinel is absent,
    // and the Reserve below says so authoritatively.
    if !s.register(std::process::id()) {
        tracing::debug!("sentinel: Register not acknowledged on {}", s.pipe());
    }
    match s.reserve(vram_mib, ram_mib, None) {
        Decision::Absent => {
            tracing::info!("sentinel: absent — starting the engine without a reservation");
            Ok(Grant::absent())
        }
        Decision::Granted(r) => {
            tracing::info!(free_mib = r.free_mib, "sentinel: reserved {vram_mib} MiB VRAM + {ram_mib} MiB RAM");
            Ok(Grant::held(s))
        }
        Decision::Refused(r) => {
            let message = refusal_message(vram_mib, ram_mib, &r);
            if forced() {
                eprintln!("{message}\n(--force given: starting anyway)");
                Ok(Grant::held(s))
            } else {
                Err(Refused {
                    message,
                    free_mib: r.free_mib,
                    blockers: r.blockers.unwrap_or_default(),
                })
            }
        }
    }
}

/// The text a refused engine start prints. Pure, so a test can pin it.
pub fn refusal_message(vram_mib: u64, ram_mib: u64, r: &Reserved) -> String {
    let mut s = format!("Sentinel refused the engine: asked for {vram_mib} MiB VRAM + {ram_mib} MiB RAM");
    match r.free_mib {
        Some(f) => s.push_str(&format!(", {f} MiB VRAM free")),
        None => s.push_str(", free VRAM unknown"),
    }
    if let Some(f) = r.ram_free_mib { s.push_str(&format!(", {f} MiB RAM free")); }
    s.push('.');
    if let Some(reason) = &r.reason { s.push_str(&format!("\n  reason: {reason}")); }
    match r.blockers.as_deref() {
        None => s.push_str("\n  blockers: not reported by this Sentinel build"),
        Some([]) => s.push_str(
            "\n  holding it: nobody — no other client has a booking, the card is simply this full",
        ),
        Some(list) => {
            s.push_str("\n  holding it:");
            for b in list {
                s.push_str(&format!("\n    {b}"));
            }
        }
    }
    s.push_str("\nsay 'pause batch jobs' to Dexter or stop the holder, then run this again (or pass --force).");
    s
}

// --------------------------------------------------------------------------------------------
// events that mean "put the engine down"
// --------------------------------------------------------------------------------------------

/// Does this event tell a serving engine to stop? Returns the line to print.
///
/// Two do: VRAM Red — freezing does not free VRAM under WDDM, so the only
/// cooperative move left is to exit — and a `Preempt` addressed to us.
/// Everything else (Yellow, relief, heartbeats, someone else's preemption) is
/// ignored.
pub fn stop_reason(event: &Value, client: &str) -> Option<String> {
    let Value::Object(m) = event else { return None };
    let (key, body) = m.iter().next()?;
    match key.as_str() {
        "VramTight" => {
            let level = body.get("level").and_then(Value::as_str).unwrap_or("");
            if !level.eq_ignore_ascii_case("red") { return None; }
            match body.get("free_mib").and_then(Value::as_u64) {
                Some(f) => Some(format!("Sentinel: VRAM is Red ({f} MiB free) — stopping the engine so the card recovers")),
                None => Some("Sentinel: VRAM is Red — stopping the engine so the card recovers".to_string()),
            }
        }
        "Preempt" => {
            // `client` names the target. An old service may omit it; an
            // unaddressed Preempt is treated as ours, since we are the batch
            // holder it would be aimed at.
            if let Some(t) = body.get("client").and_then(Value::as_str) {
                if t != client { return None; }
            }
            let by = body.get("by").and_then(Value::as_str).unwrap_or("another client");
            match body.get("mib").and_then(Value::as_u64) {
                Some(mib) => Some(format!("Sentinel: preempted by {by}, which needs {mib} MiB — stopping the engine (re-run the job when it is done)")),
                None => Some(format!("Sentinel: preempted by {by} — stopping the engine (re-run the job when it is done)")),
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
