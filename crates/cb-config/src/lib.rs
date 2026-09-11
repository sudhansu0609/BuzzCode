//! Configuration schema + layered loading for buzzcode.
//!
//! Layers (later overrides earlier):
//!   1. built-in defaults
//!   2. `~/.buzzcode/config.toml`
//!   3. `<project>/.buzzcode/config.toml`
//!   4. `BUZZCODE_*` environment variables (a few well-known ones)
//!   5. CLI overrides (applied by the binary)
//!
//! TOML tables are deep-merged; `[[profile]]` arrays are merged by `name`.

mod merge;
mod paths;
mod schema;

pub use paths::Paths;
pub use schema::*;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Load the effective configuration for `project_dir`.
pub fn load(project_dir: &Path) -> Result<Config> {
    let paths = Paths::discover(project_dir)?;
    let mut doc = toml::Value::try_from(Config::default())?;

    for file in [&paths.user_config, &paths.project_config] {
        if file.is_file() {
            let text = std::fs::read_to_string(file)
                .with_context(|| format!("reading {}", file.display()))?;
            let layer: toml::Value = toml::from_str(&text)
                .with_context(|| format!("parsing {}", file.display()))?;
            merge::merge_into(&mut doc, layer);
            tracing::debug!(file = %file.display(), "config layer applied");
        }
    }

    merge::apply_env(&mut doc);

    let mut cfg: Config = doc.try_into().context("validating merged config")?;
    cfg.paths = paths;
    cfg.finalize()?;
    Ok(cfg)
}

/// Save `profile` into the user config (`~/.buzzcode/config.toml`) as a `[[profile]]` entry
/// (replacing one with the same name) and optionally make it the default.
pub fn persist_profile(cfg: &Config, profile: &Profile, make_default: bool) -> Result<PathBuf> {
    let path = cfg.paths.user_config.clone();
    let mut doc: toml::Value = match std::fs::read_to_string(&path) {
        Ok(t) => toml::from_str(&t).with_context(|| format!("parsing {}", path.display()))?,
        Err(_) => toml::Value::Table(Default::default()),
    };
    // Only store fields that differ from defaults to keep the file readable.
    let full = toml::Value::try_from(profile)?;
    let defaults = toml::Value::try_from(Profile::default())?;
    let mut entry = toml::map::Map::new();
    if let (toml::Value::Table(f), toml::Value::Table(d)) = (&full, &defaults) {
        for (k, v) in f {
            if k == "name" || k == "model" || k == "alias" || d.get(k) != Some(v) { entry.insert(k.clone(), v.clone()); }
        }
    }
    let root = doc.as_table_mut().context("config root is not a table")?;
    let profiles = root.entry("profile").or_insert_with(|| toml::Value::Array(vec![]));
    if let toml::Value::Array(arr) = profiles {
        arr.retain(|p| p.get("name").and_then(toml::Value::as_str) != Some(&profile.name));
        arr.push(toml::Value::Table(entry));
    }
    if make_default {
        let general = root.entry("general").or_insert_with(|| toml::Value::Table(Default::default()));
        if let toml::Value::Table(g) = general { g.insert("default_profile".into(), toml::Value::String(profile.name.clone())); }
    }
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::write(&path, toml::to_string_pretty(&doc)?)?;
    Ok(path)
}

/// Persist a permission-mode choice in the user config.
pub fn persist_permission_mode(cfg: &Config, mode: &str) -> Result<PathBuf> {
    let path = cfg.paths.user_config.clone();
    let mut doc: toml::Value = match std::fs::read_to_string(&path) { Ok(t) => toml::from_str(&t)?, Err(_) => toml::Value::Table(Default::default()) };
    let root = doc.as_table_mut().context("config root is not a table")?;
    let general = root.entry("general").or_insert_with(|| toml::Value::Table(Default::default()));
    if let toml::Value::Table(g) = general { g.insert("permission_mode".into(), toml::Value::String(mode.into())); }
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::write(&path, toml::to_string_pretty(&doc)?)?;
    Ok(path)
}

/// Render the built-in default config as TOML (for `buzzcode config init`).
pub fn default_toml() -> String {
    toml::to_string_pretty(&Config::default()).expect("default config serializes")
}

/// Expand `~` and environment variables in a path string.
pub fn expand_path(s: &str) -> PathBuf {
    let mut out = s.to_string();
    if let Some(rest) = out.strip_prefix("~/").or_else(|| out.strip_prefix("~\\")) {
        if let Some(home) = directories::BaseDirs::new().map(|b| b.home_dir().to_path_buf()) {
            out = home.join(rest).to_string_lossy().into_owned();
        }
    } else if out == "~" {
        if let Some(home) = directories::BaseDirs::new().map(|b| b.home_dir().to_path_buf()) {
            out = home.to_string_lossy().into_owned();
        }
    }
    // %VAR% (Windows) and $VAR / ${VAR}
    let mut result = String::with_capacity(out.len());
    let mut chars = out.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '%' => {
                let mut name = String::new();
                let mut closed = false;
                for n in chars.by_ref() {
                    if n == '%' { closed = true; break; }
                    name.push(n);
                }
                if closed {
                    result.push_str(&std::env::var(&name).unwrap_or_default());
                } else {
                    result.push('%');
                    result.push_str(&name);
                }
            }
            '$' => {
                let braced = chars.peek() == Some(&'{');
                if braced { chars.next(); }
                let mut name = String::new();
                while let Some(&n) = chars.peek() {
                    if braced {
                        if n == '}' { chars.next(); break; }
                        name.push(n); chars.next();
                    } else if n.is_alphanumeric() || n == '_' {
                        name.push(n); chars.next();
                    } else { break; }
                }
                if name.is_empty() { result.push('$'); }
                else { result.push_str(&std::env::var(&name).unwrap_or_default()); }
            }
            _ => result.push(c),
        }
    }
    PathBuf::from(result)
}
