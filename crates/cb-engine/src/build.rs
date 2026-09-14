//! Build llama.cpp from source (delegates to scripts/build-llama.ps1 on Windows or build-llama.sh on macOS)
//! or fetch a prebuilt release from GitHub.

use anyhow::{bail, Context, Result};
use cb_config::Config;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Locate build script relative to the executable or the workspace.
/// Checks `scripts/`, `scripts/windows/`, and `scripts/macos/`.
fn find_script(name: &str) -> Option<PathBuf> {
    let mut cands = Vec::new();
    let subdirs = ["scripts", "scripts/windows", "scripts/macos"];
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.clone();
        for _ in 0..4 {
            if let Some(parent) = p.parent() {
                p = parent.to_path_buf();
                for sub in subdirs { cands.push(p.join(sub).join(name)); }
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for sub in subdirs { cands.push(cwd.join(sub).join(name)); }
    }
    for sub in subdirs {
        cands.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../").join(sub).join(name));
    }
    cands.into_iter().find(|p| p.is_file())
}

pub struct BuildOptions {
    pub pin: Option<String>,
    pub clean: bool,
    pub allow_cuda_132: bool,
    pub allow_unsupported_compiler: bool,
}

/// Run the Windows PowerShell build script, streaming output through `on_line`.
#[cfg(windows)]
pub async fn build_from_source(cfg: &Config, opts: &BuildOptions, on_line: impl Fn(&str)) -> Result<PathBuf> {
    let script = find_script("build-llama.ps1")
        .context("scripts/build-llama.ps1 (or scripts/windows/build-llama.ps1) not found")?;
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

/// Run the macOS / Unix shell build script, streaming output through `on_line`.
#[cfg(not(windows))]
pub async fn build_from_source(cfg: &Config, opts: &BuildOptions, on_line: impl Fn(&str)) -> Result<PathBuf> {
    let script = find_script("build-llama.sh")
        .context("scripts/macos/build-llama.sh not found")?;
    let mut cmd = tokio::process::Command::new("bash");
    cmd.arg(&script);
    cmd.arg(cb_config::expand_path(&cfg.engine.llama_cpp_dir));
    let pin = opts.pin.clone().unwrap_or_else(|| cfg.engine.pin.clone());
    if !pin.is_empty() { cmd.arg(pin); }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().context("spawning bash build script")?;
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

/// Download a prebuilt release archive from GitHub into ~/.buzzcode/engine/prebuilt.
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

    let (bin, cudart) = if cfg!(windows) {
        // Prefer CUDA 13.x win x64 binaries, then 12.x.
        let is_bin = |n: &str, ver: &str| n.starts_with("llama-") && n.contains("bin-win-cuda") && n.contains(ver) && n.ends_with("-x64.zip");
        let bin = pick(&|n| is_bin(n, "13."))
            .or_else(|| pick(&|n| is_bin(n, "12.")))
            .context("no Windows CUDA asset in release")?;
        let cudart = pick(&|n| n.starts_with("cudart-llama-bin-win-cuda") && n.contains(if bin.0.contains("13.") { "13." } else { "12." }) && n.ends_with("-x64.zip"));
        (bin, cudart)
    } else if cfg!(target_os = "macos") {
        // Prefer Apple Silicon Metal archive, fallback to any macos archive
        let is_mac_arm = |n: &str| n.starts_with("llama-") && n.contains("bin-macos-arm64");
        let is_mac_any = |n: &str| n.starts_with("llama-") && n.contains("bin-macos");
        let bin = pick(&is_mac_arm)
            .or_else(|| pick(&is_mac_any))
            .context("no macOS asset found in llama.cpp release")?;
        (bin, None)
    } else {
        let is_linux = |n: &str| n.starts_with("llama-") && (n.contains("bin-ubuntu-x64") || n.contains("bin-linux-x64"));
        let bin = pick(&is_linux).context("no Linux release asset found")?;
        (bin, None)
    };

    let dir = cfg.paths.engine_dir().join("prebuilt");
    tokio::fs::create_dir_all(&dir).await?;
    for (name, url) in std::iter::once(bin.clone()).chain(cudart.into_iter()) {
        let archive = dir.join(&name);
        on_line(&format!("downloading {name}"));
        crate::download::download_url(&url, &archive, None).await?;
        on_line(&format!("extracting {name}"));
        extract_to(&archive, &dir).await?;
    }

    let exe_name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    let exe = dir.join(exe_name);
    if !exe.is_file() {
        // In tarballs, it might be in a subfolder or directly in dir
        let found = find_binary_in_dir(&dir, exe_name);
        if let Some(f) = found {
            let _ = std::fs::rename(&f, &exe);
        }
    }
    if !exe.is_file() { bail!("{exe_name} missing after extraction"); }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&exe) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&exe, perms);
        }
    }

    std::fs::write(cfg.paths.engine_dir().join("prebuilt.txt"), format!("{tag}\n{}\n", bin.0))?;
    Ok(exe)
}

fn find_binary_in_dir(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.file_name().map(|n| n == name).unwrap_or(false) {
                return Some(p);
            } else if p.is_dir() {
                if let Some(found) = find_binary_in_dir(&p, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

async fn extract_to(archive: &Path, dir: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        let status = tokio::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command"])
            .arg(format!("Expand-Archive -LiteralPath '{}' -DestinationPath '{}' -Force", archive.display(), dir.display()))
            .status().await?;
        if !status.success() { bail!("Expand-Archive failed for {}", archive.display()); }
    }
    #[cfg(not(windows))]
    {
        let path_str = archive.to_string_lossy();
        let status = if path_str.ends_with(".tar.gz") || path_str.ends_with(".tgz") {
            tokio::process::Command::new("tar")
                .args(["-xzf"]).arg(archive).arg("-C").arg(dir)
                .status().await?
        } else {
            tokio::process::Command::new("unzip")
                .args(["-o"]).arg(archive).arg("-d").arg(dir)
                .status().await?
        };
        if !status.success() { bail!("Extraction failed for {}", archive.display()); }
    }
    Ok(())
}
