//! Persisted per-model tuning results (`~/.buzzcode/tune/<hash>.toml`).

use crate::vram::VramPlan;
use anyhow::Result;
use cb_config::Config;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TunedProfile {
    pub model_file: String,
    pub model_hash: String,
    pub profile: String,
    pub ngl: u32,
    pub n_ctx: u32,
    pub override_tensor: Option<String>,
    pub ffn_cpu_blocks: u32,
    pub threads: Option<u32>,
    pub spec: Option<String>,
    pub measured_decode_tps: Option<f64>,
    pub measured_prompt_tps: Option<f64>,
    pub measured_vram_mib: Option<u64>,
    pub free_vram_mib_at_tune: u64,
    pub tuned_at: String,
    /// Number of consecutive successful starts with this plan (confidence).
    pub successes: u32,
}

/// Cheap identity for a model file: size + mtime + first/last 1 MiB hashed (full hash of 15 GB is slow).
pub fn model_hash(path: &Path) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let meta = f.metadata()?;
    let mut h = blake3::Hasher::new();
    h.update(&meta.len().to_le_bytes());
    let mut buf = vec![0u8; 1 << 20];
    let n = f.read(&mut buf)?;
    h.update(&buf[..n]);
    if meta.len() > (2 << 20) {
        f.seek(SeekFrom::End(-(1 << 20)))?;
        let n = f.read(&mut buf)?;
        h.update(&buf[..n]);
    }
    Ok(h.finalize().to_hex()[..16].to_string())
}

fn tune_path(cfg: &Config, hash: &str, profile: &str) -> PathBuf {
    cfg.paths.tune_dir().join(format!("{hash}-{profile}.toml"))
}

pub fn load(cfg: &Config, hash: &str, profile: &str) -> Option<TunedProfile> {
    let p = tune_path(cfg, hash, profile);
    let text = std::fs::read_to_string(p).ok()?;
    toml::from_str(&text).ok()
}

pub fn save(cfg: &Config, t: &TunedProfile) -> Result<()> {
    std::fs::create_dir_all(cfg.paths.tune_dir())?;
    let p = tune_path(cfg, &t.model_hash, &t.profile);
    std::fs::write(p, toml::to_string_pretty(t)?)?;
    Ok(())
}

impl TunedProfile {
    pub fn from_plan(model: &Path, hash: &str, profile: &str, plan: &VramPlan) -> Self {
        Self {
            model_file: model.to_string_lossy().into_owned(),
            model_hash: hash.into(),
            profile: profile.into(),
            ngl: plan.ngl,
            n_ctx: plan.n_ctx,
            override_tensor: plan.override_tensor.clone(),
            ffn_cpu_blocks: plan.ffn_cpu_blocks,
            free_vram_mib_at_tune: plan.free_vram_bytes >> 20,
            tuned_at: humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
            ..Default::default()
        }
    }

    /// A tuned plan is reusable if free VRAM hasn't dropped by more than ~256 MiB since tuning.
    pub fn still_valid(&self, free_vram_mib_now: u64) -> bool {
        free_vram_mib_now + 256 >= self.free_vram_mib_at_tune
    }

    pub fn to_plan(&self) -> VramPlan {
        VramPlan {
            ngl: self.ngl, n_ctx: self.n_ctx, override_tensor: self.override_tensor.clone(),
            ffn_cpu_blocks: self.ffn_cpu_blocks, ctx_checkpoints: 8,
            est_vram_bytes: self.measured_vram_mib.unwrap_or(0) << 20, est_cpu_weight_bytes: 0,
            free_vram_bytes: self.free_vram_mib_at_tune << 20,
            rationale: format!("restored tuned profile from {}", self.tuned_at),
        }
    }
}
