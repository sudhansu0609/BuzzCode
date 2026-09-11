//! GPU memory probe via `nvidia-smi` (no NVML binding needed).

use anyhow::{Context, Result};
use std::process::Command;

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct GpuMem {
    pub total_mib: u64,
    pub used_mib: u64,
    pub free_mib: u64,
}

impl GpuMem {
    pub fn total_bytes(&self) -> u64 { self.total_mib << 20 }
    pub fn free_bytes(&self) -> u64 { self.free_mib << 20 }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub driver: String,
    pub mem: GpuMem,
}

pub fn query() -> Result<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .context("running nvidia-smi (is the NVIDIA driver installed?)")?;
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

/// Free memory only (cheap poll used during autotune / OOM retries).
pub fn free_mib() -> Result<u64> { Ok(query()?.mem.free_mib) }

/// Total physical RAM and available RAM in bytes (Windows via PowerShell/CIM; fallback to sysinfo-free heuristics).
pub fn system_ram() -> (u64, u64) {
    #[cfg(windows)]
    {
        let out = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command",
                "$o=Get-CimInstance Win32_OperatingSystem; \"$($o.TotalVisibleMemorySize) $($o.FreePhysicalMemory)\""])
            .output();
        if let Ok(o) = out {
            let s = String::from_utf8_lossy(&o.stdout);
            let mut it = s.split_whitespace();
            if let (Some(t), Some(f)) = (it.next(), it.next()) {
                if let (Ok(t), Ok(f)) = (t.parse::<u64>(), f.parse::<u64>()) {
                    return (t * 1024, f * 1024);
                }
            }
        }
    }
    (32 << 30, 16 << 30)
}
