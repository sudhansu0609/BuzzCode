//! `buzzcode models [use <sel>]` — rank models for this machine and pick one.

use anyhow::{bail, Context, Result};
use cb_config::Config;
use cb_engine::catalog;
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// Make a model the default (by number from the list, profile name, file name, or path).
    Use { selection: String },
    /// Pick the recommended model automatically and make it the default.
    Auto,
}

pub fn list(cfg: &Config) -> Result<()> {
    let (gpu, cands) = catalog::recommend(cfg)?;
    print!("{}", catalog::render_table(&gpu, &cands, Some(&cfg.general.default_profile)));
    println!("\nswitch: `buzzcode models use <#|name>` · auto: `buzzcode models auto` · one-off: `buzzcode --model <#|name>`");
    Ok(())
}

pub fn run(cfg: Config, cmd: Option<ModelsCmd>) -> Result<()> {
    match cmd {
        None => list(&cfg),
        Some(ModelsCmd::Auto) => {
            let (gpu, cands) = catalog::recommend(&cfg)?;
            let best = cands.first().context("no models found")?;
            let p = catalog::profile_for(&cfg, best);
            let path = cb_config::persist_profile(&cfg, &p, true)?;
            println!("{}", catalog::render_table(&gpu, &cands[..cands.len().min(5)], None));
            println!("default model → {} ({}) saved in {}", p.name, best.note, path.display());
            if !best.is_local() { println!("it will be downloaded on first start (`buzzcode engine doctor`)."); }
            Ok(())
        }
        Some(ModelsCmd::Use { selection }) => {
            let (_, cands) = catalog::recommend(&cfg)?;
            if let Some(c) = catalog::select(&cands, &cfg, &selection) {
                let p = catalog::profile_for(&cfg, c);
                let path = cb_config::persist_profile(&cfg, &p, true)?;
                println!("default model → {} · {} · est {:.0} tok/s · saved in {}", p.name, c.note, c.est_tps, path.display());
                if !c.fits_gpu { println!("note: this model does not fully fit in free VRAM; expect slower decode."); }
                return Ok(());
            }
            if let Some(p) = cfg.profile(&selection).cloned() {
                let path = cb_config::persist_profile(&cfg, &p, true)?;
                println!("default model → {} · saved in {}", p.name, path.display());
                return Ok(());
            }
            if cfg.active_profile().is_external(&cfg.engine) || cfg.engine.provider == cb_config::EngineProvider::External {
                let mut p = cfg.active_profile().clone();
                p.model = selection.clone();
                p.alias = selection.clone();
                let path = cb_config::persist_profile(&cfg, &p, true)?;
                println!("default model → {} (model: {}) · saved in {}", p.name, selection, path.display());
                return Ok(());
            }
            bail!("no model matches {selection:?}; run `buzzcode models`")
        }
    }
}

/// Apply `--model` for this run only.
pub fn apply_override(cfg: &mut Config, selection: &str) -> Result<()> {
    if cfg.profile(selection).is_some() { cfg.general.default_profile = selection.to_string(); return Ok(()); }
    let (_, cands) = catalog::recommend(cfg)?;
    if let Some(c) = catalog::select(&cands, cfg, selection) {
        let p = catalog::profile_for(cfg, c);
        let name = p.name.clone();
        cfg.profiles.retain(|x| x.name != name);
        cfg.profiles.push(p);
        cfg.general.default_profile = name;
        return Ok(());
    }
    if cfg.active_profile().is_external(&cfg.engine) || cfg.engine.provider == cb_config::EngineProvider::External {
        let mut p = cfg.active_profile().clone();
        p.model = selection.to_string();
        p.alias = selection.to_string();
        let name = format!("{}-{}", p.name, selection.replace('/', "-"));
        p.name = name.clone();
        cfg.profiles.push(p);
        cfg.general.default_profile = name;
        return Ok(());
    }
    bail!("no model matches {selection:?}; run `buzzcode models`")
}
