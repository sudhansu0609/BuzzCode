use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Well-known locations. Populated by [`crate::load`]; the serde derive exists only so
/// `Config` can carry it (it is skipped during (de)serialization).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Paths {
    /// `~/.buzzcode`
    pub user_dir: PathBuf,
    /// `~/.buzzcode/config.toml`
    pub user_config: PathBuf,
    /// Project root (where `.buzzcode/` lives). Defaults to the cwd passed in.
    pub project_dir: PathBuf,
    /// `<project>/.buzzcode`
    pub project_state_dir: PathBuf,
    /// `<project>/.buzzcode/config.toml`
    pub project_config: PathBuf,
}

impl Paths {
    pub fn discover(project_dir: &Path) -> Result<Self> {
        let base = directories::BaseDirs::new().context("cannot resolve home directory")?;
        let user_dir = base.home_dir().join(".buzzcode");
        let project_dir = std::path::absolute(project_dir)
            .with_context(|| format!("resolving {}", project_dir.display()))?;
        let project_state_dir = project_dir.join(".buzzcode");
        Ok(Self {
            user_config: user_dir.join("config.toml"),
            project_config: project_state_dir.join("config.toml"),
            user_dir,
            project_dir,
            project_state_dir,
        })
    }

    pub fn logs_dir(&self) -> PathBuf { self.user_dir.join("logs") }
    pub fn slots_dir(&self) -> PathBuf { self.user_dir.join("slots") }
    pub fn engine_dir(&self) -> PathBuf { self.user_dir.join("engine") }
    pub fn tune_dir(&self) -> PathBuf { self.user_dir.join("tune") }
    pub fn tool_out_dir(&self) -> PathBuf { self.project_state_dir.join("tool-out") }
    pub fn plans_dir(&self) -> PathBuf { self.project_state_dir.join("plans") }
    pub fn bench_dir(&self) -> PathBuf { self.project_state_dir.join("bench") }

    /// Create every directory we may write to.
    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [
            self.user_dir.clone(), self.logs_dir(), self.slots_dir(), self.engine_dir(),
            self.tune_dir(), self.project_state_dir.clone(), self.tool_out_dir(),
            self.plans_dir(), self.bench_dir(),
        ] {
            std::fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
        }
        Ok(())
    }
}
