//! Hugging Face Hub downloads with HTTP Range resume and optional sha256 verification.

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone)]
pub struct Progress {
    pub downloaded: u64,
    pub total: Option<u64>,
}

pub type ProgressFn = Box<dyn Fn(Progress) + Send + Sync>;

/// Download `file` from HF `repo` into `dest_dir/<repo>/<file>`. Returns the final path.
/// Split files (`*-00001-of-0000N.gguf`) are handled by the caller (download each shard).
pub async fn hf_download(repo: &str, file: &str, dest_dir: &Path, on_progress: Option<ProgressFn>) -> Result<PathBuf> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let dest = dest_dir.join(repo).join(file);
    download_url(&url, &dest, on_progress).await?;
    Ok(dest)
}

pub async fn download_url(url: &str, dest: &Path, on_progress: Option<ProgressFn>) -> Result<()> {
    if let Some(parent) = dest.parent() { tokio::fs::create_dir_all(parent).await?; }
    let part = dest.with_extension(format!("{}.part", dest.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default()));
    let existing = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);

    let client = reqwest::Client::builder()
        .user_agent("buzzcode/0.1 (+https://github.com/buzzcaf/buzzcode)")
        .build()?;

    // HEAD for size (follows redirects to the CDN).
    let head = client.head(url).send().await.context("HEAD request")?;
    if !head.status().is_success() { bail!("HEAD {url} → {}", head.status()); }
    let total = head.headers().get(reqwest::header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<u64>().ok());

    if let Some(t) = total {
        if dest.is_file() && tokio::fs::metadata(dest).await.map(|m| m.len()).unwrap_or(0) == t {
            tracing::info!(path = %dest.display(), "already downloaded");
            return Ok(());
        }
        if existing == t {
            tokio::fs::rename(&part, dest).await?;
            return Ok(());
        }
        if existing > t {
            tokio::fs::remove_file(&part).await.ok();
        }
    }

    let start = if total.map(|t| existing < t).unwrap_or(false) { existing } else { 0 };
    let mut req = client.get(url);
    if start > 0 { req = req.header(reqwest::header::RANGE, format!("bytes={start}-")); }
    let resp = req.send().await.context("GET request")?;
    let status = resp.status();
    if !(status.is_success() || status == reqwest::StatusCode::PARTIAL_CONTENT) { bail!("GET {url} → {status}"); }
    let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT && start > 0;

    let mut file = tokio::fs::OpenOptions::new()
        .create(true).write(true).append(resumed).truncate(!resumed)
        .open(&part).await?;
    let mut downloaded = if resumed { start } else { 0 };
    let mut stream = resp.bytes_stream();
    let mut last_report = std::time::Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream chunk")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if let Some(cb) = &on_progress {
            if last_report.elapsed().as_millis() > 200 {
                cb(Progress { downloaded, total });
                last_report = std::time::Instant::now();
            }
        }
    }
    file.flush().await?;
    drop(file);
    if let Some(cb) = &on_progress { cb(Progress { downloaded, total }); }
    if let Some(t) = total {
        if downloaded != t { bail!("size mismatch: got {downloaded}, expected {t}"); }
    }
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

/// sha256 of a file (streaming). Used when the model card publishes a digest.
pub async fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path).await?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    let digest = h.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Expand a split-GGUF filename into all shards (`name-00001-of-00003.gguf` → 3 names).
pub fn shard_names(file: &str) -> Vec<String> {
    if let Some(idx) = file.find("-of-") {
        if idx >= 6 {
            let pre = &file[..idx - 5];
            let post = &file[idx + 4..];
            let n_str: String = post.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = n_str.parse::<u32>() {
                let tail = &post[n_str.len()..];
                return (1..=n).map(|i| format!("{pre}{i:05}-of-{n_str}{tail}")).collect();
            }
        }
    }
    vec![file.to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shards() {
        let v = shard_names("Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf");
        assert_eq!(v, vec!["Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf", "Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf"]);
        assert_eq!(shard_names("x.gguf"), vec!["x.gguf"]);
    }
}
