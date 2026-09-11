//! Unit tests for the Sentinel client, driven by a **real** named pipe.
//!
//! There is no way to fake `\\.\pipe\…` at the `std::fs` layer, and mocking the
//! transport would leave the one thing most likely to break — the framing —
//! untested. So each test stands up a throwaway pipe server of its own with
//! `CreateNamedPipeW`, scripted with the exact lines the service would send.
//!
//! Nothing here ever touches `\\.\pipe\Sentinel`: every client is built with
//! `Sentinel::with_pipe`, so the live service on this machine is not involved.

use super::*;
use serde_json::json;

// --------------------------------------------------------------------------------------------
// pure helpers
// --------------------------------------------------------------------------------------------

#[test]
fn pipe_path_defaults_and_honours_the_env_override() {
    assert_eq!(resolve_pipe(None), DEFAULT_PIPE);
    assert_eq!(resolve_pipe(Some("   ")), DEFAULT_PIPE);
    assert_eq!(resolve_pipe(Some(r"\\.\pipe\SentinelDev")), r"\\.\pipe\SentinelDev");
}

#[test]
fn red_vram_stops_the_engine_but_yellow_does_not() {
    let red = json!({"VramTight": {"level": "red", "free_mib": 512}});
    let msg = stop_reason(&red, CLIENT).expect("red must stop the engine");
    assert!(msg.contains("512 MiB free"), "{msg}");
    assert!(stop_reason(&json!({"VramTight": {"level": "yellow", "free_mib": 3000}}), CLIENT).is_none());
    assert!(stop_reason(&json!({"VramRelief": {"free_mib": 9000}}), CLIENT).is_none());
    assert!(stop_reason(&json!({"Heartbeat": {}}), CLIENT).is_none());
}

#[test]
fn preempt_is_obeyed_only_when_it_names_us() {
    let ours = json!({"Preempt": {"client": "buzzcode", "by": "dexter", "mib": 5120, "deadline_secs": 20}});
    let msg = stop_reason(&ours, CLIENT).expect("a Preempt aimed at us must stop the engine");
    assert!(msg.contains("dexter") && msg.contains("5120"), "{msg}");

    let theirs = json!({"Preempt": {"client": "vrdeep", "by": "dexter", "mib": 5120}});
    assert!(stop_reason(&theirs, CLIENT).is_none(), "someone else's preemption is not ours");

    // An old service that does not address the event: we are the batch holder.
    let unaddressed = json!({"Preempt": {"by": "dexter"}});
    assert!(stop_reason(&unaddressed, CLIENT).is_some());
}

#[test]
fn a_refusal_names_the_free_vram_the_holders_and_the_way_out() {
    let r = Reserved {
        granted: false,
        free_mib: Some(1800),
        ram_free_mib: Some(4096),
        reason: Some("not enough free VRAM".into()),
        blockers: Some(vec![Blocker {
            process: Some("VrDeep".into()),
            pid: Some(4242),
            vram_mib: Some(10240),
            class: Some("batch".into()),
            evictable: Some(false),
            ..Default::default()
        }]),
        ..Default::default()
    };
    let m = refusal_message(12500, 6500, &r);
    assert!(m.contains("1800 MiB VRAM free"), "{m}");
    assert!(m.contains("VrDeep (pid 4242) 10240 MiB VRAM, batch"), "{m}");
    assert!(m.contains("not enough free VRAM"), "{m}");
    assert!(m.contains("say 'pause batch jobs' to Dexter or stop the holder"), "{m}");
}

#[test]
fn an_old_service_without_blockers_still_produces_a_usable_message() {
    let r = Reserved { granted: false, free_mib: Some(900), ..Default::default() };
    let m = refusal_message(12500, 6500, &r);
    assert!(m.contains("900 MiB VRAM free"), "{m}");
    assert!(m.contains("not reported by this Sentinel build"), "{m}");
    assert!(m.contains("say 'pause batch jobs'"), "{m}");
}

/// "Nobody is holding a booking" and "this build does not tell us who is" are
/// different facts, and only one of them is about the build.
///
/// Found live: against the **v2** service, a refusal that was purely about free
/// VRAM printed "blockers: not reported by this Sentinel build" — untrue, and
/// it sends the owner looking for a Sentinel upgrade instead of for the memory.
/// A v2 reply always carries the key, so an empty list is an answer.
#[test]
fn a_v2_refusal_with_nobody_holding_a_booking_does_not_blame_the_build() {
    let r = Reserved {
        granted: false,
        free_mib: Some(7786),
        ram_free_mib: Some(17867),
        reason: Some("not enough VRAM: 7602 MiB requested but only 3786 MiB available".into()),
        blockers: Some(vec![]),
        ..Default::default()
    };
    let m = refusal_message(7602, 12224, &r);
    assert!(!m.contains("not reported by this Sentinel build"), "{m}");
    assert!(m.contains("nobody"), "{m}");
    assert!(m.contains("7786 MiB VRAM free"), "{m}");
    assert!(m.contains("say 'pause batch jobs'"), "{m}");
}

// --------------------------------------------------------------------------------------------
// over a real pipe
// --------------------------------------------------------------------------------------------

#[cfg(windows)]
mod over_a_pipe {
    use super::*;
    use fake::FakeSentinel;

    /// `--force` is a process-wide switch, so the two tests that flip it take
    /// turns rather than racing each other.
    static FORCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn absent_sentinel_means_go_ahead() {
        // Nothing is listening on this name and nothing ever will be.
        let s = Sentinel::with_pipe(format!(r"\\.\pipe\buzzcode-not-there-{}", std::process::id()));
        assert!(matches!(s.reserve(12500, 6500, None), Decision::Absent));
        assert!(!s.register(1234));
        // gate_with must let the engine start regardless.
        let g = gate_with(s, 12500, 6500).expect("an absent guardian never blocks a start");
        assert!(g.absent);
    }

    #[test]
    fn register_accepts_the_old_ok_reply() {
        let fake = FakeSentinel::start(1, |_req| vec![json!("Ok")]);
        let s = Sentinel::with_pipe(fake.name());
        assert!(s.register(4711));
        let req = fake.requests().remove(0);
        let body = &req["Register"];
        assert_eq!(body["client"], "buzzcode");
        assert_eq!(body["pid"], 4711);
        assert_eq!(body["class"], "batch");
        assert_eq!(body["priority"], 30);
        assert_eq!(body["label"], "buzzcode engine");
    }

    #[test]
    fn register_accepts_the_v2_registered_reply() {
        let fake = FakeSentinel::start(1, |_req| vec![json!({"Registered": {"token": "abc123"}})]);
        assert!(Sentinel::with_pipe(fake.name()).register(4711));
    }

    #[test]
    fn register_rejects_anything_else() {
        let fake = FakeSentinel::start(1, |_req| vec![json!({"Error": "NotOwner"})]);
        assert!(!Sentinel::with_pipe(fake.name()).register(4711));
    }

    #[test]
    fn a_granted_reserve_carries_the_free_vram() {
        let fake = FakeSentinel::start(1, |_req| {
            vec![json!({"Reserved": {"granted": true, "free_mib": 14000, "reserved_mib": 12500,
                                     "ram_free_mib": 20000, "expires_in_secs": 90, "reason": "ok"}})]
        });
        let s = Sentinel::with_pipe(fake.name());
        match s.reserve(12500, 6500, None) {
            Decision::Granted(r) => {
                assert_eq!(r.free_mib, Some(14000));
                assert_eq!(r.reserved_mib, Some(12500));
            }
            other => panic!("expected Granted, got {other:?}"),
        }
        let body = fake.requests().remove(0)["Reserve"].clone();
        assert_eq!(body["client"], "buzzcode");
        assert_eq!(body["mib"], 12500);
        assert_eq!(body["ram_mib"], 6500);
        assert!(body.get("for_pid").is_none(), "for_pid is omitted before the server exists");
    }

    #[test]
    fn for_pid_is_sent_once_the_server_pid_is_known() {
        let fake = FakeSentinel::start(1, |_req| vec![json!({"Reserved": {"granted": true}})]);
        let s = Sentinel::with_pipe(fake.name());
        let _ = s.reserve(12500, 6500, Some(9090));
        assert_eq!(fake.requests().remove(0)["Reserve"]["for_pid"], 9090);
    }

    #[test]
    fn a_refused_reserve_stops_the_gate_with_the_blockers_named() {
        let _guard = FORCE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Three connections: the direct reserve below, then the gate's
        // Register and Reserve.
        let fake = FakeSentinel::start(3, |_req| {
            vec![json!({"Reserved": {
                "granted": false, "free_mib": 1800, "ram_free_mib": 4096,
                "reason": "not enough free VRAM",
                "blockers": [{"process": "VrDeep", "pid": 4242, "vram_mib": 10240,
                              "class": "batch", "evictable": false}]
            }})]
        });
        let s = Sentinel::with_pipe(fake.name());
        match s.reserve(12500, 6500, None) {
            Decision::Refused(r) => {
                assert_eq!(r.free_mib, Some(1800));
                let blockers = r.blockers.clone().expect("a v2 reply carries the key");
                assert_eq!(blockers.len(), 1);
                assert_eq!(blockers[0].process.as_deref(), Some("VrDeep"));
            }
            other => panic!("expected Refused, got {other:?}"),
        }
        // …and the gate turns that into an error the CLI can exit 3 on.
        set_force(false);
        let err = gate_with(Sentinel::with_pipe(fake.name()), 12500, 6500)
            .err()
            .expect("a refusal must stop the start");
        assert!(err.message.contains("VrDeep"), "{}", err.message);
        assert_eq!(err.free_mib, Some(1800));
    }

    #[test]
    fn force_starts_the_engine_over_a_refusal() {
        let _guard = FORCE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Register, Reserve, and the Release the returned Grant sends on drop.
        let fake = FakeSentinel::start(3, |req| {
            if req.get("Register").is_some() { vec![json!("Ok")] }
            else { vec![json!({"Reserved": {"granted": false, "free_mib": 100}})] }
        });
        set_force(true);
        let got = gate_with(Sentinel::with_pipe(fake.name()), 12500, 6500);
        set_force(false);
        let grant = got.expect("--force must win");
        assert!(!grant.absent, "the booking is still held, so Release is sent on drop");
    }

    #[test]
    fn a_slow_service_is_abandoned_and_treated_as_absent() {
        let fake = FakeSentinel::start_slow(1, std::time::Duration::from_millis(1500));
        let s = Sentinel::with_pipe(fake.name());
        let t0 = std::time::Instant::now();
        let d = s.reserve(12500, 6500, None);
        let elapsed = t0.elapsed();
        assert!(matches!(d, Decision::Absent), "a slow guardian is an absent guardian");
        assert!(elapsed < std::time::Duration::from_millis(900), "call took {elapsed:?}, must be ~300 ms");
    }

    #[test]
    fn release_sends_the_client_name() {
        let fake = FakeSentinel::start(1, |_req| vec![json!("Ok")]);
        Sentinel::with_pipe(fake.name()).release();
        assert_eq!(fake.requests().remove(0)["Release"]["client"], "buzzcode");
    }

    #[test]
    fn subscribe_streams_events_to_the_handler() {
        let fake = FakeSentinel::start(1, |_req| {
            vec![
                json!("Subscribed"),
                json!({"Heartbeat": {"at": 1}}),
                json!({"VramTight": {"level": "yellow", "free_mib": 3000}}),
                json!({"VramTight": {"level": "red", "free_mib": 400}}),
            ]
        });
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let sub = Sentinel::with_pipe(fake.name()).subscribe(move |ev| {
            if let Some(reason) = stop_reason(&ev, CLIENT) { let _ = tx.send(reason); }
        });
        let reason = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the red event must reach the handler");
        assert!(reason.contains("400 MiB free"), "{reason}");
        sub.stop();
        assert_eq!(fake.requests().remove(0)["Subscribe"]["client"], "buzzcode");
    }
}

// --------------------------------------------------------------------------------------------
// the fake service
// --------------------------------------------------------------------------------------------

#[cfg(windows)]
mod fake {
    use serde_json::Value;
    use std::io::{Read, Write};
    use std::os::windows::io::FromRawHandle;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Storage::FileSystem::PIPE_ACCESS_DUPLEX;
    use windows_sys::Win32::System::Pipes::{
        ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, PIPE_READMODE_BYTE, PIPE_TYPE_BYTE,
        PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
    };

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// A scripted stand-in for the Sentinel service on a pipe of its own.
    ///
    /// Serves `max_conns` connections then goes away, so a test can never leave
    /// a thread parked in `ConnectNamedPipe` forever.
    pub struct FakeSentinel {
        name: String,
        requests: Arc<Mutex<Vec<Value>>>,
    }

    impl FakeSentinel {
        pub fn name(&self) -> &str { &self.name }

        /// Every request the fake has received so far, in order.
        pub fn requests(&self) -> Vec<Value> { self.requests.lock().unwrap().clone() }

        /// `reply` maps one request line to the lines written back (a command
        /// answer is one line; a `Subscribe` answer is `"Subscribed"` plus the
        /// event stream).
        pub fn start<F>(max_conns: usize, reply: F) -> Self
        where
            F: Fn(&Value) -> Vec<Value> + Send + Sync + 'static,
        {
            Self::spawn(max_conns, Duration::ZERO, Arc::new(reply))
        }

        /// A service that answers, but far too late to be waited for.
        pub fn start_slow(max_conns: usize, delay: Duration) -> Self {
            Self::spawn(max_conns, delay, Arc::new(|_| vec![Value::String("Ok".into())]))
        }

        fn spawn(
            max_conns: usize,
            delay: Duration,
            reply: Arc<dyn Fn(&Value) -> Vec<Value> + Send + Sync>,
        ) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!(r"\\.\pipe\buzzcode-fake-sentinel-{}-{n}", std::process::id());
            let requests = Arc::new(Mutex::new(Vec::new()));
            // Every instance is created up front, before `start` returns, and
            // each gets a thread of its own. Serving them one after another
            // would leave a window with no listening instance, which the client
            // — correctly — reads as "Sentinel is absent", and the gate tests
            // need several round trips in a row.
            for _ in 0..max_conns {
                // The handle crosses to the worker as a `usize`: a raw HANDLE is
                // not `Send`.
                let handle = create_instance(&name);
                if handle == 0 { continue; }
                let (reply, reqs) = (reply.clone(), requests.clone());
                std::thread::spawn(move || {
                    unsafe { ConnectNamedPipe(handle as _, std::ptr::null_mut()) };
                    // SAFETY: the handle came from CreateNamedPipeW and is not
                    // touched again once `File` owns it.
                    let mut f = unsafe { std::fs::File::from_raw_handle(handle as *mut _) };
                    if let Some(req) = read_line(&mut f) {
                        reqs.lock().unwrap().push(req.clone());
                        if !delay.is_zero() { std::thread::sleep(delay); }
                        for line in reply(&req) {
                            let mut bytes = serde_json::to_vec(&line).unwrap();
                            bytes.push(b'\n');
                            if f.write_all(&bytes).is_err() { break; }
                            let _ = f.flush();
                        }
                    }
                    // Let the client drain what we wrote before the handle dies.
                    std::thread::sleep(Duration::from_millis(30));
                    unsafe { DisconnectNamedPipe(handle as _) };
                    drop(f); // closes the handle
                });
            }
            Self { name, requests }
        }
    }

    /// A fresh instance of the pipe, as a `usize` so it can be moved to a thread.
    fn create_instance(name: &str) -> usize {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let h = unsafe {
            CreateNamedPipeW(
                wide.as_ptr(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                PIPE_UNLIMITED_INSTANCES,
                4096,
                4096,
                0,
                std::ptr::null(),
            )
        };
        if h as isize == INVALID_HANDLE_VALUE as isize { 0 } else { h as usize }
    }

    fn read_line(f: &mut std::fs::File) -> Option<Value> {
        let mut buf = Vec::new();
        let mut one = [0u8; 1];
        loop {
            match f.read(&mut one) {
                Ok(0) => return None,
                Ok(_) => { if one[0] == b'\n' { break; } buf.push(one[0]); }
                Err(_) => return None,
            }
        }
        serde_json::from_slice(&buf).ok()
    }
}
