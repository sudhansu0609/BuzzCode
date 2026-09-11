//! `buzzcode index ...`: scan, stats, map, search, outline — without starting the engine.

use anyhow::Result;
use cb_codeintel::repomap::MapOptions;
use cb_codeintel::{Indexer, RepoMap, SymKind};
use cb_config::Config;
use clap::Subcommand;
use std::sync::Arc;

#[derive(Subcommand, Debug)]
pub enum IndexCmd {
    /// Scan the project (incremental) and print stats.
    Scan,
    /// Print the ranked repo map.
    Map { #[arg(long, default_value_t = 2048)] budget: u32, #[arg(long)] focus: Vec<String> },
    /// Fuzzy symbol search.
    Search { query: String, #[arg(long)] kind: Option<String>, #[arg(long, default_value_t = 20)] limit: usize },
    /// Outline of one file.
    Outline { path: String },
}

fn open(cfg: &Config) -> Result<Arc<Indexer>> {
    let db = cfg.paths.project_dir.join(&cfg.codeintel.index_path);
    Ok(Arc::new(Indexer::open(&cfg.paths.project_dir, &db, &cfg.codeintel)?))
}

pub fn run(cfg: Config, cmd: IndexCmd) -> Result<()> {
    let idx = open(&cfg)?;
    let st = idx.scan()?;
    match cmd {
        IndexCmd::Scan => {
            println!("files {}  parsed {}  unchanged {}  removed {}  symbols {}  in {} ms", st.files_seen, st.parsed, st.unchanged, st.removed, st.symbols, st.ms);
            for (l, n) in idx.language_summary() { println!("  {l:<12} {n}"); }
        }
        IndexCmd::Map { budget, focus } => {
            let t0 = std::time::Instant::now();
            let map = RepoMap::build(&idx);
            let est = |s: &str| (s.len() as f32 / 3.2).ceil() as u32;
            let s = map.render(&MapOptions { budget_tokens: budget, focus: &focus, mentions: &[], estimate: &est });
            println!("{s}");
            eprintln!("[{} files in graph, ~{} tokens, {} ms]", map.graph.files.len(), est(&s), t0.elapsed().as_millis());
        }
        IndexCmd::Search { query, kind, limit } => {
            for h in idx.search(&query, kind.as_deref().and_then(SymKind::parse), limit) {
                println!("{}:{} {} {}", h.path, h.symbol.line_start, h.symbol.kind.label(), h.symbol.signature);
            }
        }
        IndexCmd::Outline { path } => {
            let p = cfg.paths.project_dir.join(path);
            let syms = idx.outline(&p)?;
            print!("{}", cb_codeintel::index::format_outline(&idx.rel(&p), &syms));
        }
    }
    Ok(())
}
