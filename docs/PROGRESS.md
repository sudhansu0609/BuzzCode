# BuzzCode progress

## 2026-09-10 — P3: the engine takes a port instead of assuming one

Cross-repo context: `Buzzcaf_Media/GUARDIAN_PLAN.md` §11 ("Ports — nothing
hardcoded, step forward, publish, discover"). `engine.port = 8089` in
`config.toml` used to be a fact: `llama-server` was spawned with it, the client
was built from it, `engine.json` recorded it and `engine status` asked it. If
anything else held 8089 — a stale `llama-server`, a second checkout — the spawn
failed to bind, and every one of those four then lied about where the engine was.

### `crates/cb-engine/src/ports.rs` (new)

The Rust third of the shared helper, alongside `dexter/backend/buzzcaf_ports.py`
and `BuzzEdit/scripts/buzzcaf-ports.mjs`. Same ledger, same lock, same entry
shape:

```
ledger_path()                                  %LOCALAPPDATA%\Buzzcaf\ports.json
                                               (BUZZCAF_PORTS_FILE overrides)
pick_port(preferred, span, host) -> u16        a real bind, never a probe
publish_at(ledger, app, port, health, extra)   read-merge-write under
withdraw_at(ledger, app)                       ports.json.lock, temp + rename
entries() / entry(app) / published_port(..)
```

`bind`, not connect: a probe calls a TIME_WAIT socket free and then the spawn
fails anyway. Nothing here is fatal — a ledger that cannot be written is a
discovery problem, never a reason the engine will not serve.

### Wiring

- `EngineManager` gained `port: Mutex<u16>` and turned `client` into an
  `RwLock<LlamaClient>`. `start()` calls `pick_port(cfg.engine.port, 20, host)`
  before it resolves the binary, and `set_port()` rebuilds the client, so the
  spawned `--port` and every later request agree by construction. `client()`
  returns a clone rather than a borrow now, because the base URL can move
  between starts — and because a read guard must never be held across an
  `.await` (the crash watcher spawns `start()` onto the runtime).
- `LlamaServerArgs::build(..)` takes the port explicitly instead of reading
  `cfg.engine.port`. That is what keeps the command line and the client from
  drifting apart.
- Publish on Ready, withdraw on `stop()`:
  `{"buzzcode": {port, pid, started_at, health, extra: {alias, model}}}`.
  A `published` flag means `stop()` withdraws only what `start()` published.
  The `External` provider publishes nothing — that engine is not ours.
- `engine.json` carries `mgr.port()`, not `cfg.engine.port`.
- `engine status` resolves through the new `live_base_url()`: `engine.json`
  first, then the ledger, then the configured wish. `llama-server`'s `/health`
  does not name itself, so those two files are the only honest sources and the
  identity scan the plan describes elsewhere does not apply here.
- `engine serve` prints the real URL, and says so when it had to step forward.

### Evidence

- `cargo test -p cb-engine -p buzzcode` — **45 passed, 0 failed** (was 31; the
  14 new ones are `ports::tests::*`, every one against a temp ledger under
  `BUZZCAF_PORTS_FILE`, so the machine's real one is never touched). They cover:
  a free port taken as is; a busy port stepped over *with its holder still
  listening afterwards*; an exhausted window falling back to an OS port; the
  entry shape; a merge that keeps another app's row; a re-publish replacing only
  our own; an idempotent withdraw; a corrupt ledger and a BOM'd ledger both
  rebuilt rather than fatal; a stale lock broken and released; eight threads
  publishing at once and all eight surviving.
- `cargo build --release -p buzzcode` — clean, 2m 05s.
  `C:\Users\singh\.cargo\bin\buzzcode.exe` was **not** touched; the new binary is
  `target/release/buzzcode.exe`.
- Live, with a dummy `/health` responder standing in for the engine so the real
  27B never started (VrDeep is on the card):
  - ledger says 8091 → `engine http://127.0.0.1:8091 : healthy`
  - empty ledger → falls back to `engine http://127.0.0.1:8089 : DOWN`
  - `engine.json` says 8092 while the ledger says 8091 → status reports 8092,
    i.e. the record wins, and `engine stop` acts on that same record.
    (That `engine.json` was written for the check and removed again; there was
    none on this machine.)

**Not verified end to end:** the publish/withdraw pair inside a real
`engine serve`. Reaching it means spawning the 27B, which the plan forbids while
VrDeep is rendering. What rests on review is the two call sites in `manager.rs`
— `publish_port()` after `/health` answers and `withdraw_port()` at the top of
`stop()`; the helper underneath them is covered by the 14 tests, and
`engine status` was exercised against real ledger and record files. A hard kill
still leaves a stale entry, which is by design: readers check that the pid is
alive and that the health route names the app.

---

## 2026-09-10 — BC1 + BC2: the engine can be stopped by name, and asks before it takes the card

Cross-repo context: `Buzzcaf_Media/GUARDIAN_PLAN.md` §5. Sentinel is the
machine's guardian of VRAM and RAM; buzzcode's 27B engine is the single biggest
claimant on a 16 GB card, and until now it was also the only one that could
neither be asked to leave nor be found from another process.

### BC1 — `~/.buzzcode/engine.json`, `engine stop`, `engine restart`

`crates/cb-engine/src/pidfile.rs` (new). `engine serve` writes

```json
{ "pid": 12345, "server_pid": 23456, "port": 8089,
  "model": "Qwen3.8-27B-UD-IQ3_XXS.gguf", "alias": "qwen3.8-27b",
  "started_at": "2026-09-10T16:02:11Z" }
```

as soon as `/health` answers, and removes it on the way out (Ctrl-C, or a stop
ordered by Sentinel). The write goes through a `.tmp` sibling and a rename, so a
reader never sees half a document; the reader also skips a UTF-8 BOM, because a
file that has been looked at in Notepad or rewritten by PowerShell's
`Set-Content -Encoding utf8` has one.

- **`engine stop`** kills the `llama-server` tree first (`taskkill /T /F`) and
  the supervising `buzzcode` second — the other order orphans the server with
  all of its VRAM still held — then removes the record and exits 0. Nothing
  recorded, or a record whose PIDs are both gone, exits **1** with a message
  (and a stale record is cleared on the way out).
- **`engine restart`** is `stop` that tolerates "nothing was running", a 1.5 s
  pause for Windows to release port 8089, then `serve`.
- **`engine status`** prints the record — with a live/GONE mark per PID — before
  the existing `/health` + `/props` + `/slots` probe. The record is deliberately
  first: it is the only thing that still answers when the server has wedged and
  `/health` never returns.

The file is advisory. A hard kill leaves it behind, and that is a normal
outcome, not an error.

### BC2 — `crates/cb-engine/src/sentinel.rs`

A ~340-line named-pipe JSON client for `\\.\pipe\Sentinel` (`SENTINEL_PIPE`
overrides the path; the dev service uses `\\.\pipe\SentinelDev`). Newline
delimited, serde's externally-tagged shapes, no new runtime dependency.

Three properties it is built around:

1. **Absent Sentinel = yesterday's behaviour.** Not installed, not running,
   slow, wrong ACL, unparsable answer — every one of those reads as "go ahead".
   `BUZZCODE_SENTINEL=0` turns the client off entirely.
2. **Nothing blocks.** Windows named-pipe I/O has no per-call timeout, so each
   request runs on a worker thread that is abandoned after **300 ms**.
3. **A refusal is a refusal.** `--force` is the owner's override, not ours.

Commands spoken: `Register` (`client":"buzzcode"`, `class":"batch"`,
`priority":30`, `label":"buzzcode engine"` — both the old `"Ok"` and G10's
`{"Registered":{token}}` are accepted as an ack), `Reserve` (`mib` = the
planner's estimated total VRAM, `ram_mib` = `engine.cache_ram_mb` + the weights
the plan spills to host, `for_pid` when known), `Release`, `Subscribe`.
`Reserved.blockers` is optional — the build installed today does not send it,
and the refusal message says so rather than pretending there are no holders.

Where it hooks in: **`EngineManager::start()`**, immediately after the VRAM plan
is computed and before the first spawn. That one place covers `engine serve`,
the implicit engine start behind `buzzcode -p …`, and the TUI, because all three
come through it. The grant is released the moment `/health` answers — from then
on the server's real usage is visible to Sentinel's own accounting — and `Grant`
releases on drop, so an early `?` cannot leak a booking.

A refusal aborts the start with a message naming the free VRAM, the blockers and
the way out:

```
Sentinel refused the engine: asked for 12500 MiB VRAM + 6500 MiB RAM, 1800 MiB VRAM free.
  reason: not enough free VRAM
  holding it:
    VrDeep (pid 4242) 10240 MiB VRAM, batch
say 'pause batch jobs' to Dexter or stop the holder, then run this again (or pass --force).
```

`main()` recognises that error and exits **3**, so a caller can tell "the card is
spoken for" apart from a real failure. `--force` (global flag) prints the same
message and starts anyway.

`engine serve` also subscribes to the event stream *before* it starts loading, so
an order to give the card back that arrives during a slow model load is not lost.
`{"VramTight":{"level":"red",…}}` and a `Preempt` addressed to `buzzcode` both
stop the server and print why. Yellow, relief, heartbeats and someone else's
preemption are ignored — freezing does not free VRAM under WDDM, so exiting is
the only cooperative move left at Red, and it is too blunt a move to make at
Yellow.

### Verification

| Command | Result |
|---|---|
| `cargo build --release -p buzzcode` | `Finished \`release\` profile ... in 2m 03s` |
| `cargo test -p cb-engine -p buzzcode` | `30 passed; 0 failed` (cb-engine), `0 passed` (buzzcode has no tests) |

The 17 new tests are in `sentinel/tests.rs` and `pidfile.rs`. There is no way to
fake `\\.\pipe\…` at the `std::fs` layer, and mocking the transport would leave
the framing — the thing most likely to break — untested, so each test stands up
a **real** throwaway pipe server of its own with `CreateNamedPipeW`
(`windows-sys`, already in `Cargo.lock`, added as a `cfg(windows)`
dev-dependency). Every client is built with `Sentinel::with_pipe`, so the live
service on this machine is never involved. Covered: both `Register` acks; the
`Reserve` wire shape with and without `for_pid`; a grant; a refusal with and
without `blockers`; the gate turning a refusal into an error and `--force`
turning it back into a start; a 1.5 s-slow service being abandoned in ~300 ms; an
unreachable pipe reading as "go ahead"; `Release`; and a `Subscribe` stream whose
red event reaches the handler.

Exercised live against the release binary (no engine started):

- `engine stop` with no record → the message, exit 1.
- `engine status` / `engine stop` against a record naming two live dummy
  processes → both listed as running, then `stopped llama-server (pid 29452) and
  buzzcode engine serve (pid 9416)`, exit 0, both processes gone, second `stop`
  exits 1.
- A record naming two dead PIDs → `status` marks both GONE and points at
  `engine stop`; `stop` removes the stale record and exits 1.

**Not verified end to end:** the exit-3 path was not exercised against a live
`buzzcode engine serve`, because doing so means reaching the gate, which is
after model resolution — and a fake pipe that failed to answer in 300 ms would
have read as "Sentinel absent" and spawned the real 27B onto a GPU that VrDeep
is rendering on. The refusal itself is covered by unit tests
(`a_refused_reserve_stops_the_gate_with_the_blockers_named`); only the
`downcast_ref::<sentinel::Refused>()` → `exit(3)` line in `main.rs` rests on
review. Likewise the live `Register`/`Reserve` round trip against the installed
G8 service is untested: an old service that cannot parse the new optional fields
answers nothing, which the client reads as absent, so the failure mode is
"start as before", never a block.

`C:\Users\singh\.cargo\bin\buzzcode.exe` was **not** touched; the new binary is
`target/release/buzzcode.exe`.
