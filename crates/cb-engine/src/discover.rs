//! Locate `llama-server.exe` and GGUF model files.

use anyhow::{bail, Context, Result};
use cb_config::Config;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct BuildInfo {
    pub dir: String,
    pub pin: String,
    pub commit: String,
    pub cuda_path: String,
    pub cuda_arch: String,
    pub server: String,
    #[serde(default)]
    pub built_at: String,
}

pub fn read_build_info(cfg: &Config) -> Option<BuildInfo> {
    let p = cfg.paths.engine_dir().join("build-info.toml");
    let text = std::fs::read_to_string(p).ok()?;
    toml::from_str(&text).ok()
}

/// Find llama-server: explicit config → build-info.toml → llama_cpp_dir/build → ~/.buzzcode/engine/prebuilt → PATH.
pub fn find_server(cfg: &Config) -> Result<PathBuf> {
    let exe = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    let mut candidates: Vec<PathBuf> = Vec::new();

    if !cfg.engine.server_binary.is_empty() {
        candidates.push(cb_config::expand_path(&cfg.engine.server_binary));
    }
    let prebuilt = cfg.paths.engine_dir().join("prebuilt").join(exe);
    if cfg.engine.provider == cb_config::EngineProvider::Prebuilt { candidates.push(prebuilt.clone()); }
    if let Some(bi) = read_build_info(cfg) {
        candidates.push(PathBuf::from(bi.server));
    }
    let src = cb_config::expand_path(&cfg.engine.llama_cpp_dir);
    for sub in ["build/bin/Release", "build/bin", "build"] {
        candidates.push(src.join(sub).join(exe));
    }
    candidates.push(prebuilt);
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            candidates.push(dir.join(exe));
        }
    }
    for c in &candidates {
        if c.is_file() { return Ok(c.clone()); }
    }
    bail!(
        "llama-server not found. Run `buzzcode engine build` (or scripts\\build-llama.ps1), set engine.server_binary, \
         or set engine.provider = \"external\" with external_url."
    )
}

/// Resolve a profile `model` string to a local GGUF path, if present locally.
/// Accepts: absolute path, relative path (under models_dir / extra dirs), bare filename (searched recursively),
/// or `repo:file` (file part searched locally under models_dir/<repo>/ and recursively).
pub fn find_model(cfg: &Config, model: &str) -> Option<PathBuf> {
    let direct = cb_config::expand_path(model);
    if direct.is_absolute() && direct.is_file() { return Some(direct); }

    let (repo, file) = split_repo_file(model);
    let mut roots = vec![cfg.engine.models_dir_path()];
    roots.extend(cfg.engine.extra_search_paths());
    roots.push(cfg.paths.project_dir.clone());

    // 1. direct relative path under roots
    for r in &roots {
        let p = r.join(&direct);
        if p.is_file() { return Some(p); }
        if let Some(repo) = repo {
            let p = r.join(repo).join(file);
            if p.is_file() { return Some(p); }
        }
    }
    // 2. recursive search by filename (depth-limited)
    let fname = Path::new(file).file_name()?.to_string_lossy().to_string();
    for r in &roots {
        if let Some(p) = find_file_recursive(r, &fname, 4) { return Some(p); }
    }
    None
}

pub fn split_repo_file(model: &str) -> (Option<&str>, &str) {
    // "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-IQ3_XXS.gguf" → (repo, file); Windows drive letters (C:\) are not repos.
    if let Some((a, b)) = model.split_once(':') {
        if a.contains('/') && !a.contains('\\') && a.len() > 2 {
            return (Some(a), b);
        }
    }
    (None, model)
}

fn find_file_recursive(root: &Path, fname: &str, depth: u32) -> Option<PathBuf> {
    if depth == 0 || !root.is_dir() { return None; }
    let rd = std::fs::read_dir(root).ok()?;
    let mut dirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file() {
            if p.file_name().map(|n| n.to_string_lossy().eq_ignore_ascii_case(fname)).unwrap_or(false) { return Some(p); }
        } else if p.is_dir() {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name.starts_with('.') || name == "target" || name == "node_modules" { continue; }
            dirs.push(p);
        }
    }
    for d in dirs {
        if let Some(p) = find_file_recursive(&d, fname, depth - 1) { return Some(p); }
    }
    None
}

/// List all GGUF files under the configured roots (for `buzzcode engine models`).
pub fn list_local_models(cfg: &Config) -> Vec<(PathBuf, u64)> {
    let mut roots = vec![cfg.engine.models_dir_path()];
    roots.extend(cfg.engine.extra_search_paths());
    let mut out = Vec::new();
    fn walk(dir: &Path, depth: u32, out: &mut Vec<(PathBuf, u64)>) {
        if depth == 0 { return; }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { walk(&p, depth - 1, out); }
            else if p.extension().map(|x| x.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
                let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                out.push((p, size));
            }
        }
    }
    for r in roots { walk(&r, 5, &mut out); }
    out.sort();
    out.dedup();
    out
}

/// Split-GGUF helper: given any shard, return the first shard (which has the full metadata).
pub fn first_shard(path: &Path) -> PathBuf {
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    // Qwen3.8-27B-UD-Q4_K_XL-00002-of-00003.gguf → -00001-of-00003
    if let Some(idx) = name.find("-of-") {
        if idx >= 6 {
            let pre = &name[..idx - 5];
            let post = &name[idx..];
            return path.with_file_name(format!("{pre}00001{post}"));
        }
    }
    path.to_path_buf()
}

pub fn context_check(cfg: &Config) -> Result<()> {
    let _ = find_server(cfg).context("engine binary")?;
    Ok(())
}
