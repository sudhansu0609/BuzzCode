//! Build llama.cpp from source (delegates to scripts/build-llama.ps1) or fetch a prebuilt release.

use anyhow::{bail, Context, Result};
use cb_config::Config;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Locate scripts/build-llama.ps1 relative to the executable or the workspace.
fn find_script(name: &str) -> Option<PathBuf> {
    let mut cands = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.clone();
        for _ in 0..4 { if let Some(parent) = p.parent() { p = parent.to_path_buf(); cands.push(p.join("scripts").join(name)); } }
    }
    if let Ok(cwd) = std::env::current_dir() { cands.push(cwd.join("scripts").join(name)); }
    cands.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts").join(name));
    cands.into_iter().find(|p| p.is_file())
}

pub struct BuildOptions {
    pub pin: Option<String>,
    pub clean: bool,
    pub allow_cuda_132: bool,
    pub allow_unsupported_compiler: bool,
}

/// Run the PowerShell build script, streaming its output through `on_line`.
pub async fn build_from_source(cfg: &Config, opts: &BuildOptions, on_line: impl Fn(&str)) -> Result<PathBuf> {
    let script = find_script("build-llama.ps1").context("scripts/build-llama.ps1 not found")?;
    let mut cmd = tokio::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]).arg(&script);
    cmd.arg("-Dir").arg(cb_config::expand_path(&cfg.engine.llama_cpp_dir));
    let pin = opts.pin.clone().unwrap_or_else(|| cfg.engine.pin.clone());
    if !pin.is_empty() { cmd.arg("-Pin").arg(pin); }
    if !cfg.engine.cuda_path.is_empty() { cmd.arg("-CudaPath").arg(cb_config::expand_path(&cfg.engine.cuda_path)); }
    cmd.arg("-CudaArch").arg(&cfg.engine.cuda_arch);
    if opts.clean { cmd.arg("-Clean"); }
    if opts.allow_cuda_132 { cmd.arg("-AllowCuda132"); }
    if opts.allow_unsupported_compiler { cmd.arg("-AllowUnsupportedCompiler"); }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().context("spawning powershell")?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let mut out = BufReader::new(stdout).lines();
    let mut err = BufReader::new(stderr).lines();
    loop {
        tokio::select! {
            l = out.next_line() => match l? { Some(l) => on_line(&l), None => break },
            l = err.next_line() => if let Some(l) = l? { on_line(&l) },
        }
    }
    while let Some(l) = err.next_line().await? { on_line(&l); }
    let status = child.wait().await?;
    if !status.success() { bail!("build script failed with {status}"); }
    crate::discover::find_server(cfg)
}

/// Download a prebuilt Windows CUDA release zip (+ cudart) from GitHub into ~/.buzzcode/engine/prebuilt.
pub async fn fetch_prebuilt(cfg: &Config, tag: Option<&str>, on_line: impl Fn(&str)) -> Result<PathBuf> {
    let client = reqwest::Client::builder().user_agent("buzzcode/0.1").build()?;
    let tag = match tag {
        Some(t) => t.to_string(),
        None => {
            let v: serde_json::Value = client.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").send().await?.json().await?;
            v["tag_name"].as_str().context("no tag_name")?.to_string()
        }
    };
    on_line(&format!("release {tag}"));
    let rel: serde_json::Value = client.get(format!("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")).send().await?.json().await?;
    let assets = rel["assets"].as_array().context("no assets")?;
    let pick = |pred: &dyn Fn(&str) -> bool| -> Option<(String, String)> {
        assets.iter().find_map(|a| {
            let name = a["name"].as_str()?;
            if pred(name) { Some((name.to_string(), a["browser_download_url"].as_str()?.to_string())) } else { None }
        })
    };
    // Prefer CUDA 13.x win x64 binaries, then 12.x.
    let is_bin = |n: &str, ver: &str| n.starts_with("llama-") && n.contains("bin-win-cuda") && n.contains(ver) && n.ends_with("-x64.zip");
    let bin = pick(&|n| is_bin(n, "13."))
        .or_else(|| pick(&|n| is_bin(n, "12.")))
        .context("no Windows CUDA asset in release")?;
    let cudart = pick(&|n| n.starts_with("cudart-llama-bin-win-cuda") && n.contains(if bin.0.contains("13.") { "13." } else { "12." }) && n.ends_with("-x64.zip"));

    let dir = cfg.paths.engine_dir().join("prebuilt");
    tokio::fs::create_dir_all(&dir).await?;
    for (name, url) in std::iter::once(bin.clone()).chain(cudart.into_iter()) {
        let zip = dir.join(&name);
        on_line(&format!("downloading {name}"));
        crate::download::download_url(&url, &zip, None).await?;
        on_line(&format!("extracting {name}"));
        unzip_to(&zip, &dir).await?;
    }
    let exe = dir.join("llama-server.exe");
    if !exe.is_file() { bail!("llama-server.exe missing after extraction"); }
    std::fs::write(cfg.paths.engine_dir().join("prebuilt.txt"), format!("{tag}\n{}\n", bin.0))?;
    Ok(exe)
}

async fn unzip_to(zip: &Path, dir: &Path) -> Result<()> {
    // Use PowerShell's Expand-Archive (avoids a zip crate dependency).
    let status = tokio::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command"])
        .arg(format!("Expand-Archive -LiteralPath '{}' -DestinationPath '{}' -Force", zip.display(), dir.display()))
        .status().await?;
    if !status.success() { bail!("Expand-Archive failed for {}", zip.display()); }
    Ok(())
}
