//! Generic Linux / Unix GPU and memory probe.

use super::types::{GpuInfo, GpuMem};
use anyhow::{Context, Result};
use std::process::Command;

pub fn query() -> Result<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .context("running nvidia-smi")?;
    if !out.status.success() {
        anyhow::bail!("nvidia-smi failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.lines().next().unwrap_or("").trim();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 5 { anyhow::bail!("unexpected nvidia-smi output: {line:?}"); }
    let p = |i: usize| parts[i].parse::<u64>().with_context(|| format!("parsing nvidia-smi field {i}: {:?}", parts[i]));
    Ok(GpuInfo {
        name: parts[0].to_string(),
        driver: parts[1].to_string(),
        mem: GpuMem { total_mib: p(2)?, used_mib: p(3)?, free_mib: p(4)? },
    })
}

pub fn free_mib() -> Result<u64> { Ok(query()?.mem.free_mib) }

pub fn system_ram() -> (u64, u64) {
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        let mut total_kb = 0u64;
        let mut avail_kb = 0u64;
        for line in text.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("MemAvailable:") {
                avail_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
        }
        if total_kb > 0 {
            return (total_kb * 1024, avail_kb * 1024);
        }
    }
    (32 << 30, 16 << 30)
}
