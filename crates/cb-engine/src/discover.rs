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
    #[cfg(target_os = "macos")]
    {
        candidates.push(PathBuf::from("/opt/homebrew/bin/llama-server"));
        candidates.push(PathBuf::from("/usr/local/bin/llama-server"));
    }
    for c in &candidates {
        if c.is_file() { return Ok(c.clone()); }
    }
    bail!(
        "llama-server not found. Run `buzzcode engine build` (or on macOS: `brew install llama.cpp`), set engine.server_binary, \
         or set engine.provider = \"external\" with external_url."
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ModelOrigin {
    LmStudio,
    Ollama,
    Local,
}

impl std::fmt::Display for ModelOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelOrigin::LmStudio => write!(f, "lmstudio"),
            ModelOrigin::Ollama => write!(f, "ollama"),
            ModelOrigin::Local => write!(f, "local"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub path: PathBuf,
    pub size: u64,
    pub display_name: String,
    pub alias: String,
    pub origin: ModelOrigin,
}

pub fn ollama_model_roots() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(var) = std::env::var("OLLAMA_MODELS") {
        let p = cb_config::expand_path(&var);
        if p.is_dir() { dirs.push(p); }
    }
    let default_home = cb_config::expand_path("~/.ollama/models");
    if default_home.is_dir() { dirs.push(default_home); }
    if cfg!(windows) {
        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            let p = PathBuf::from(userprofile).join(".ollama").join("models");
            if p.is_dir() { dirs.push(p); }
        }
        if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
            let p = PathBuf::from(localappdata).join("Ollama").join("models");
            if p.is_dir() { dirs.push(p); }
        }
    }
    if cfg!(unix) {
        let p = PathBuf::from("/usr/share/ollama/.ollama/models");
        if p.is_dir() { dirs.push(p); }
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

pub fn discover_ollama_models() -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    for root in ollama_model_roots() {
        let manifests_dir = root.join("manifests");
        let blobs_dir = root.join("blobs");
        if !manifests_dir.is_dir() || !blobs_dir.is_dir() { continue; }

        let mut manifest_files = Vec::new();
        fn walk_manifests(dir: &Path, files: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { walk_manifests(&p, files); }
                else if p.is_file() { files.push(p); }
            }
        }
        walk_manifests(&manifests_dir, &mut manifest_files);

        for mf in manifest_files {
            let Ok(rel) = mf.strip_prefix(&manifests_dir) else { continue };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            let tag_name = if let Some(stripped) = rel_str.strip_prefix("registry.ollama.ai/library/") {
                stripped.to_string()
            } else if let Some(stripped) = rel_str.strip_prefix("registry.ollama.ai/") {
                stripped.to_string()
            } else {
                rel_str.clone()
            };
            let model_tag = if let Some((m, t)) = tag_name.rsplit_once('/') {
                format!("{m}:{t}")
            } else {
                tag_name
            };

            let Ok(text) = std::fs::read_to_string(&mf) else { continue };
            let Ok(json): Result<serde_json::Value, _> = serde_json::from_str(&text) else { continue };
            let layers = json.get("layers").and_then(|v| v.as_array());
            let Some(layers) = layers else { continue };

            let model_layer = layers.iter().find(|l| {
                l.get("mediaType").and_then(|m| m.as_str()).map(|m| m.contains("model")).unwrap_or(false)
            });
            let Some(model_layer) = model_layer else { continue };
            let digest = model_layer.get("digest").and_then(|d| d.as_str()).unwrap_or_default();
            let blob_name = digest.replace(':', "-");
            let blob_path = blobs_dir.join(&blob_name);
            if !blob_path.is_file() { continue; }

            let size = blob_path.metadata().map(|m| m.len()).unwrap_or_else(|_| {
                model_layer.get("size").and_then(|s| s.as_u64()).unwrap_or(0)
            });

            if models.iter().any(|m: &DiscoveredModel| m.path == blob_path) { continue; }

            models.push(DiscoveredModel {
                path: blob_path,
                size,
                display_name: model_tag.clone(),
                alias: model_tag,
                origin: ModelOrigin::Ollama,
            });
        }
    }
    models
}

pub fn lmstudio_model_roots(cfg: &Config) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    dirs.extend(cfg.engine.extra_search_paths());
    dirs.push(cb_config::expand_path("~/.lmstudio/models"));
    dirs.push(cb_config::expand_path("~/.cache/lm-studio/models"));
    if cfg!(windows) {
        if let Ok(up) = std::env::var("USERPROFILE") {
            dirs.push(PathBuf::from(up.clone()).join(".lmstudio").join("models"));
            dirs.push(PathBuf::from(up).join(".cache").join("lm-studio").join("models"));
        }
    }
    let mut valid = Vec::new();
    for d in dirs {
        if d.is_dir() && !valid.iter().any(|v| *v == d) {
            valid.push(d);
        }
    }
    valid
}

pub fn discover_lmstudio_models(cfg: &Config) -> Vec<DiscoveredModel> {
    let roots = lmstudio_model_roots(cfg);
    let mut models = Vec::new();
    fn walk_lm(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
        if depth == 0 { return; }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                if name.starts_with('.') || name == "blobs" || name == "manifests" { continue; }
                walk_lm(&p, depth - 1, out);
            } else if p.extension().map(|x| x.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
                out.push(p);
            }
        }
    }

    let mut ggufs = Vec::new();
    for r in &roots { walk_lm(r, 6, &mut ggufs); }
    for path in ggufs {
        let fname = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let fl = fname.to_ascii_lowercase();
        if fl.contains("mmproj") || fl.starts_with("mtp-") || fl.contains("-mtp-") { continue; }
        if fname.contains("-of-") && !fname.contains("-00001-of-") { continue; }
        if models.iter().any(|m: &DiscoveredModel| m.path == path) { continue; }
        let size = path.metadata().map(|m| m.len()).unwrap_or(0);
        let total = if fname.contains("-00001-of-") { size * shard_count(&fname).max(1) } else { size };
        models.push(DiscoveredModel {
            path,
            size: total,
            display_name: fname.clone(),
            alias: fname.trim_end_matches(".gguf").to_string(),
            origin: ModelOrigin::LmStudio,
        });
    }
    models
}

pub fn discover_all_available_models(cfg: &Config) -> Vec<DiscoveredModel> {
    let mut out = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();

    // 1. Ollama downloads
    for m in discover_ollama_models() {
        if seen_paths.insert(m.path.clone()) {
            out.push(m);
        }
    }

    // 2. LM Studio downloads
    for m in discover_lmstudio_models(cfg) {
        if seen_paths.insert(m.path.clone()) {
            out.push(m);
        }
    }

    // 3. Project / engine models_dir
    let models_dir = cfg.engine.models_dir_path();
    if models_dir.is_dir() {
        let mut local_files = Vec::new();
        fn walk_local(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
            if depth == 0 { return; }
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { walk_local(&p, depth - 1, out); }
                else if p.extension().map(|x| x.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
                    out.push(p);
                }
            }
        }
        walk_local(&models_dir, 5, &mut local_files);
        for path in local_files {
            let fname = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let fl = fname.to_ascii_lowercase();
            if fl.contains("mmproj") || fl.starts_with("mtp-") || fl.contains("-mtp-") { continue; }
            if fname.contains("-of-") && !fname.contains("-00001-of-") { continue; }
            if seen_paths.insert(path.clone()) {
                let size = path.metadata().map(|m| m.len()).unwrap_or(0);
                let total = if fname.contains("-00001-of-") { size * shard_count(&fname).max(1) } else { size };
                out.push(DiscoveredModel {
                    path,
                    size: total,
                    display_name: fname.clone(),
                    alias: fname.trim_end_matches(".gguf").to_string(),
                    origin: ModelOrigin::Local,
                });
            }
        }
    }

    out
}

fn shard_count(fname: &str) -> u64 {
    fname.split("-of-").nth(1).and_then(|s| s.split('.').next()).and_then(|s| s.parse().ok()).unwrap_or(1)
}

/// Resolve a profile `model` string to a local GGUF path, if present locally.
/// Accepts: absolute path, relative path (under models_dir / extra dirs), bare filename (searched recursively),
/// Ollama model tag/name, or `repo:file` (file part searched locally under models_dir/<repo>/ and recursively).
pub fn find_model(cfg: &Config, model: &str) -> Option<PathBuf> {
    let direct = cb_config::expand_path(model);
    if direct.is_absolute() && direct.is_file() { return Some(direct); }

    // Check Ollama models by tag / name (e.g. "qwen3.5:9b", "gemma4:e4b-it-qat", "gemma-4-12b")
    for m in discover_ollama_models() {
        if m.display_name.eq_ignore_ascii_case(model)
            || m.alias.eq_ignore_ascii_case(model)
            || m.display_name.strip_suffix(":latest").map(|s| s.eq_ignore_ascii_case(model)).unwrap_or(false)
            || m.display_name.split(':').next().map(|s| s.eq_ignore_ascii_case(model)).unwrap_or(false)
        {
            return Some(m.path);
        }
    }

    // Check LM Studio models by filename or alias
    for m in discover_lmstudio_models(cfg) {
        if m.display_name.eq_ignore_ascii_case(model) || m.alias.eq_ignore_ascii_case(model) {
            return Some(m.path);
        }
    }

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

/// List all GGUF and Ollama blob model files under the configured roots and downloads.
pub fn list_local_models(cfg: &Config) -> Vec<(PathBuf, u64)> {
    let mut out: Vec<(PathBuf, u64)> = discover_all_available_models(cfg)
        .into_iter()
        .map(|m| (m.path, m.size))
        .collect();
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
