//! File-only tracing (the TUI owns the terminal). Returns a guard that must live for the process.

use cb_config::Config;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub fn init(cfg: &Config, override_filter: Option<&str>) -> tracing_appender::non_blocking::WorkerGuard {
    let filter = override_filter.map(str::to_string).unwrap_or_else(|| cfg.general.log_level.clone());
    let env_filter = EnvFilter::try_new(&filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let file_appender = tracing_appender::rolling::daily(cfg.paths.logs_dir(), "buzzcode.log");
    let (nb, guard) = tracing_appender::non_blocking(file_appender);
    let layer = fmt::layer().with_writer(nb).with_ansi(false).with_target(true).compact();
    tracing_subscriber::registry().with(env_filter).with(layer).init();
    guard
}
