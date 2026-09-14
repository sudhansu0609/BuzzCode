//! macOS GPU and Unified Memory probe (Apple Silicon Metal & sysctl).

use super::types::{GpuInfo, GpuMem};
use anyhow::Result;
use std::process::Command;

/// Probe macOS total physical RAM and available RAM in bytes.
pub fn system_ram() -> (u64, u64) {
    let total = sysctl_u64("hw.memsize").unwrap_or(16 << 30);
    let free = vm_stat_free_bytes().unwrap_or(total / 2);
    (total, free)
}

/// Query GPU / Unified Memory for macOS.
pub fn query() -> Result<GpuInfo> {
    let (total_ram, free_ram) = system_ram();
    let name = chip_name();
    
    // On Apple Silicon, GPU memory is Unified Memory.
    // By default, Metal can allocate up to ~75% of total system RAM without special entitlement.
    let total_metal_bytes = (total_ram as f64 * 0.75) as u64;
    let free_metal_bytes = free_ram.min(total_metal_bytes);
    let used_metal_bytes = total_metal_bytes.saturating_sub(free_metal_bytes);

    Ok(GpuInfo {
        name,
        driver: "Apple Metal".to_string(),
        mem: GpuMem {
            total_mib: total_metal_bytes >> 20,
            used_mib: used_metal_bytes >> 20,
            free_mib: free_metal_bytes >> 20,
        },
    })
}

/// Free memory poll used during autotune / OOM checks.
pub fn free_mib() -> Result<u64> {
    Ok(query()?.mem.free_mib)
}

fn chip_name() -> String {
    let out = Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output();
    if let Ok(o) = out {
        let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !s.is_empty() {
            if s.contains("Apple") {
                return format!("{s} (Metal Unified Memory)");
            }
            return s;
        }
    }
    "Apple Silicon (Metal)".to_string()
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.parse::<u64>().ok()
}

fn vm_stat_free_bytes() -> Option<u64> {
    let out = Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut page_size = 4096u64; // default page size
    let mut free_pages = 0u64;
    let mut inactive_pages = 0u64;
    let mut speculative_pages = 0u64;

    for line in text.lines() {
        let l = line.trim();
        if l.starts_with("Mach Virtual Memory Statistics: (page size of") {
            if let Some(bytes_str) = l.split("page size of").nth(1).and_then(|s| s.split("bytes").next()) {
                if let Ok(ps) = bytes_str.trim().parse::<u64>() {
                    page_size = ps;
                }
            }
        } else if l.starts_with("Pages free:") {
            free_pages = parse_page_count(l);
        } else if l.starts_with("Pages inactive:") {
            inactive_pages = parse_page_count(l);
        } else if l.starts_with("Pages speculative:") {
            speculative_pages = parse_page_count(l);
        }
    }

    let available_pages = free_pages + inactive_pages + speculative_pages;
    Some(available_pages * page_size)
}

fn parse_page_count(line: &str) -> u64 {
    line.split(':')
        .nth(1)
        .and_then(|s| s.trim().trim_end_matches('.').parse::<u64>().ok())
        .unwrap_or(0)
}
