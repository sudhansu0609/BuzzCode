mod bootstrap;
mod cmd_bench;
mod cmd_engine;
mod cmd_index;
mod cmd_models;
mod headless;
mod logging;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "buzzcode", version, about = "Local-model CLI coding agent (llama.cpp powered)")]
pub struct Cli {
    /// Project directory (defaults to cwd).
    #[arg(short = 'C', long, global = true)]
    pub dir: Option<PathBuf>,

    /// Model profile name from config.
    #[arg(long, global = true, env = "BUZZCODE_PROFILE")]
    pub profile: Option<String>,

    /// Model for this run: number from `buzzcode models`, profile name, file name, or GGUF path.
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Log level filter (e.g. info, debug, cb_engine=trace).
    #[arg(long, global = true)]
    pub log: Option<String>,

    /// Headless prompt: run one task and print result (no TUI).
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    /// With -p: emit JSON-lines events instead of text.
    #[arg(long)]
    pub json: bool,

    /// Permission mode: ask | edits | yolo (overrides config).
    #[arg(long, global = true)]
    pub mode: Option<String>,

    /// Preferred context size for this run, e.g. 64k or 49152 (planner shrinks it if it does not fit).
    #[arg(long, global = true)]
    pub ctx: Option<String>,

    /// Start the engine even if Sentinel refuses the VRAM/RAM reservation.
    #[arg(long, global = true)]
    pub force: bool,

    #[command(subcommand)]
    pub cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Manage the llama.cpp engine (build, doctor, models, plan, serve).
    Engine {
        #[command(subcommand)]
        cmd: cmd_engine::EngineCmd,
    },
    /// List models ranked for this machine; `models use <#|name>` / `models auto` to pick one.
    Models {
        #[command(subcommand)]
        cmd: Option<cmd_models::ModelsCmd>,
    },
    /// Open the graphical BUZZCODE ARCADE page (--demo runs it with a simulated crew, no engine).
    Arcade {
        #[arg(long)] demo: bool,
        #[arg(long)] no_open: bool,
    },
    /// Benchmarks: engine throughput, prefix-cache regression, eval tasks.
    Bench {
        #[command(subcommand)]
        cmd: cmd_bench::BenchCmd,
    },
    /// Code index: scan, map, search, outline.
    Index {
        #[command(subcommand)]
        cmd: cmd_index::IndexCmd,
    },
    /// Print the effective merged configuration.
    Config {
        /// Write the default config to ~/.buzzcode/config.toml if it does not exist.
        #[arg(long)]
        init: bool,
    },
}

/// "64k" → 65536, "49152" → 49152; rounded to a multiple of 256.
pub fn parse_ctx(s: &str) -> Result<u32> {
    let t = s.trim().to_ascii_lowercase();
    let n: u64 = if let Some(k) = t.strip_suffix('k') { k.parse::<f64>().map(|v| (v * 1024.0) as u64)? } else { t.parse()? };
    if !(2048..=1_048_576).contains(&n) { anyhow::bail!("context must be between 2k and 1024k"); }
    Ok(((n + 255) / 256 * 256) as u32)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Sentinel guards the GPU for the whole machine; `--force` is the owner
    // saying "I know, take it anyway". Set before anything can start an engine.
    cb_engine::sentinel::set_force(cli.force);
    let dir = cli.dir.clone().unwrap_or(std::env::current_dir()?);
    let mut cfg = cb_config::load(&dir)?;
    if let Some(p) = &cli.profile {
        if cfg.profile(p).is_none() { anyhow::bail!("unknown profile {p:?}"); }
        cfg.general.default_profile = p.clone();
    }
    if let Some(m) = &cli.model { cmd_models::apply_override(&mut cfg, m)?; }
    if let Some(c) = &cli.ctx {
        let n = parse_ctx(c)?;
        let name = cfg.general.default_profile.clone();
        if let Some(p) = cfg.profiles.iter_mut().find(|p| p.name == name) { p.ctx = n; p.ctx_min = p.ctx_min.min(n); }
    }
    cfg.paths.ensure_dirs()?;
    let _guard = logging::init(&cfg, cli.log.as_deref());

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().worker_threads(4).build()?;
    let outcome = rt.block_on(async move {
        match cli.cmd {
            Some(Cmd::Engine { cmd }) => cmd_engine::run(cfg, cmd).await,
            Some(Cmd::Index { cmd }) => cmd_index::run(cfg, cmd),
            Some(Cmd::Bench { cmd }) => cmd_bench::run(cfg, cmd).await,
            Some(Cmd::Models { cmd }) => cmd_models::run(cfg, cmd),
            Some(Cmd::Arcade { demo, no_open }) => {
                if demo { cb_tui::arcade_demo(cfg.tui.arcade_port.max(1), !no_open).await }
                else {
                    let url = format!("http://127.0.0.1:{}/", cfg.tui.arcade_port.max(1));
                    println!("the arcade page is served by a running `buzzcode` session at {url}; use --demo to preview without one");
                    if !no_open { let _ = std::process::Command::new("cmd").args(["/C", "start", "", &url]).spawn(); }
                    Ok(())
                }
            }
            Some(Cmd::Config { init }) => {
                if init {
                    let p = cfg.paths.user_config.clone();
                    if p.exists() { println!("exists: {}", p.display()); }
                    else { std::fs::write(&p, cb_config::default_toml())?; println!("wrote {}", p.display()); }
                } else {
                    print!("{}", toml::to_string_pretty(&cfg)?);
                }
                Ok(())
            }
            None => {
                let mode = match cli.mode.as_deref() {
                    Some("ask") => Some(cb_core::PermissionMode::Ask),
                    Some("edits") | Some("auto_accept_edits") => Some(cb_core::PermissionMode::AutoAcceptEdits),
                    Some("yolo") => Some(cb_core::PermissionMode::Yolo),
                    Some(other) => anyhow::bail!("unknown --mode {other:?} (ask|edits|yolo)"),
                    None => None,
                };
                if let Some(p) = cli.prompt {
                    headless::run(std::sync::Arc::new(cfg), p, cli.json, mode).await
                } else {
                    use std::io::IsTerminal;
                    if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
                        anyhow::bail!("the interactive TUI needs a terminal; use `buzzcode -p \"...\"` for headless runs");
                    }
                    let cfg = std::sync::Arc::new(cfg);
                    let mode = bootstrap::permission_mode(&cfg, mode);
                    eprintln!("starting engine…");
                    let built = bootstrap::build(cfg.clone(), mode, true).await?;
                    let handle = cb_core::AgentHandle::spawn(built.agent, Some(built.factory));
                    cb_tui::run(cb_tui::TuiOptions { session: built.session, handle, plans: built.plans, events_rx: built.events_rx, perm_rx: built.perm_rx.expect("interactive") }).await
                }
            }
        }
    });

    // A refused reservation is not a crash: it is the guardian saying the card
    // is spoken for. Exit 3 so a caller can tell it apart from a real failure.
    if let Err(e) = &outcome {
        if let Some(refused) = e.downcast_ref::<cb_engine::sentinel::Refused>() {
            eprintln!("{refused}");
            std::process::exit(3);
        }
    }
    outcome
}
