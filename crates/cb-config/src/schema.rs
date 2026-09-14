use crate::Paths;
use anyhow::{bail, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: General,
    pub engine: Engine,
    #[serde(rename = "profile")]
    pub profiles: Vec<Profile>,
    #[serde(rename = "model_slot")]
    pub model_slots: Vec<ModelSlot>,
    pub context: ContextCfg,
    pub codeintel: CodeIntel,
    /// glob → commands. `{file}` is substituted with the edited file path.
    pub verify: IndexMap<String, Vec<String>>,
    pub permissions: Permissions,
    #[serde(rename = "mcp_server")]
    pub mcp_servers: Vec<McpServer>,
    pub bench: Bench,
    pub tui: Tui,
    #[serde(skip)]
    pub paths: Paths,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: General::default(),
            engine: Engine::default(),
            profiles: Profile::builtin(),
            model_slots: vec![ModelSlot::default()],
            context: ContextCfg::default(),
            codeintel: CodeIntel::default(),
            verify: IndexMap::from([
                ("*.rs".to_string(), vec!["cargo check --message-format short".to_string()]),
                ("*.py".to_string(), vec!["ruff check {file}".to_string()]),
                ("*.ts".to_string(), vec!["npx tsc --noEmit -p .".to_string()]),
                ("*.tsx".to_string(), vec!["npx tsc --noEmit -p .".to_string()]),
            ]),
            permissions: Permissions::default(),
            mcp_servers: vec![],
            bench: Bench::default(),
            tui: Tui::default(),
            paths: Paths::default(),
        }
    }
}

impl Config {
    /// Post-load validation + path expansion.
    pub fn finalize(&mut self) -> Result<()> {
        if self.profiles.is_empty() { bail!("no [[profile]] defined"); }
        if !self.profiles.iter().any(|p| p.name == self.general.default_profile) {
            bail!("general.default_profile = {:?} does not match any [[profile]].name", self.general.default_profile);
        }
        for p in &mut self.profiles { p.fill_defaults(); }
        Ok(())
    }

    pub fn profile(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    pub fn active_profile(&self) -> &Profile {
        self.profile(&self.general.default_profile).expect("validated in finalize")
    }

    pub fn secondary_slot(&self) -> Option<&ModelSlot> {
        self.model_slots.iter().find(|s| s.enabled && s.role == "secondary")
    }
}

// ---------------------------------------------------------------------------
// [general]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct General {
    pub default_profile: String,
    pub permission_mode: PermissionModeCfg,
    /// Maximum turns per user request (0 = unlimited).
    pub max_turns: u32,
    pub log_level: String,
}

impl Default for General {
    fn default() -> Self {
        Self {
            default_profile: "qwen38-27b-fast".into(),
            permission_mode: PermissionModeCfg::Ask,
            max_turns: 0,
            log_level: "info".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionModeCfg { Ask, AutoAcceptEdits, Yolo }

// ---------------------------------------------------------------------------
// [engine]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Engine {
    /// `source` (we build llama.cpp), `prebuilt` (download GitHub release), `external` (connect to a running server).
    pub provider: EngineProvider,
    /// Explicit path to llama-server.exe. Overrides discovery.
    pub server_binary: String,
    pub llama_cpp_dir: String,
    /// Git tag or commit to build. Must be >= b10450 for Qwen3.8.
    pub pin: String,
    pub cuda_path: String,
    pub cuda_arch: String,
    pub host: String,
    pub port: u16,
    pub restart_max: u32,
    pub external_url: String,
    /// Optional API key for external providers (e.g. OpenRouter, DashScope, OpenAI-compatible).
    pub api_key: String,
    pub cache_ram_mb: u32,
    pub models_dir: String,
    /// Extra directories searched (recursively) for GGUF files, e.g. LM Studio's model dir.
    pub extra_search_dirs: Vec<String>,
    /// Seconds to wait for /health after spawn.
    pub startup_timeout_s: u64,
    /// Safety margin subtracted from free VRAM before planning (MiB).
    pub vram_safety_mib: u64,
    /// Reserve for the CUDA context itself (MiB).
    pub cuda_context_mib: u64,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            provider: EngineProvider::Source,
            server_binary: String::new(),
            llama_cpp_dir: "B:/llama/llama.cpp".into(),
            pin: String::new(),
            cuda_path: String::new(),
            cuda_arch: "120".into(),
            host: "127.0.0.1".into(),
            port: 8089,
            restart_max: 5,
            external_url: String::new(),
            api_key: String::new(),
            cache_ram_mb: 6144,
            models_dir: "B:/models".into(),
            extra_search_dirs: vec!["~/.lmstudio/models".into(), "~/.cache/lm-studio/models".into()],
            startup_timeout_s: 300,
            vram_safety_mib: 256,
            cuda_context_mib: 550,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineProvider { Source, Prebuilt, External }

// ---------------------------------------------------------------------------
// [[profile]]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Profile {
    pub name: String,
    /// `"<hf-repo>:<file>"` or an absolute/relative path to a .gguf (searched in models_dir + extra_search_dirs).
    pub model: String,
    pub alias: String,
    pub ctx: u32,
    pub ctx_min: u32,
    pub kv_type: String,
    pub flash_attn: bool,
    pub batch: u32,
    pub ubatch: u32,
    pub threads: u32,
    pub threads_batch: u32,
    /// "auto" or an integer.
    pub ngl: String,
    /// "auto", "" (none) or an explicit `-ot` regex list.
    pub override_tensor: String,
    /// draft-mtp | ngram-simple | none
    pub spec: String,
    pub spec_draft_n_max: u32,
    /// Path or spec to an external draft model GGUF (e.g. `mtp-Qwen3.8-Flash-Next-BF16.gguf`).
    pub draft_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<EngineProvider>,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub api_key: String,
    pub ctx_checkpoints: u32,
    pub checkpoint_min_step: u32,
    pub reasoning_format: String,
    pub chat_template_file: String,
    /// Use `--no-mmap` (recommended on Windows+CUDA when RAM allows).
    pub no_mmap: bool,
    pub extra_args: Vec<String>,
    pub sampling: SamplingSet,
    pub effort: EffortMap,
    /// Default n_predict cap per generation.
    pub max_tokens: u32,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            name: String::new(),
            model: String::new(),
            alias: "buzzcode".into(),
            // Preferred context; the planner steps down (4K at a time) to whatever fits in free VRAM, never below ctx_min.
            ctx: 65536,
            ctx_min: 32768,
            kv_type: "q8_0".into(),
            flash_attn: true,
            batch: 2048,
            ubatch: 512,
            threads: 8,
            threads_batch: 16,
            ngl: "auto".into(),
            override_tensor: "auto".into(),
            spec: "draft-mtp".into(),
            spec_draft_n_max: 2,
            draft_model: String::new(),
            provider: None,
            endpoint: String::new(),
            api_key: String::new(),
            ctx_checkpoints: 8,
            checkpoint_min_step: 2048,
            reasoning_format: "deepseek".into(),
            chat_template_file: String::new(),
            no_mmap: true,
            extra_args: vec![],
            sampling: SamplingSet::default(),
            effort: EffortMap::default(),
            max_tokens: 8192,
        }
    }
}

impl Profile {
    fn fill_defaults(&mut self) {
        if self.alias.is_empty() { self.alias = self.name.clone(); }
        if self.ctx_min > self.ctx { self.ctx_min = self.ctx; }
    }

    pub fn builtin() -> Vec<Profile> {
        vec![
            Profile {
                name: "qwen38-27b-fast".into(),
                model: "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-IQ3_XXS.gguf".into(),
                alias: "qwen3.8-27b".into(),
                ctx: 61440,
                ctx_min: 61440,
                ngl: "auto".into(),
                kv_type: "q8_0".into(),
                flash_attn: true,
                spec: "draft-mtp".into(),
                spec_draft_n_max: 2,
                no_mmap: true,
                reasoning_format: "deepseek".into(),
                ..Default::default()
            },
            Profile {
                name: "qwen38-27b-quality".into(),
                model: "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q3_K_XL.gguf".into(),
                alias: "qwen3.8-27b".into(),
                ctx: 24576,
                ..Default::default()
            },
            Profile {
                name: "qwen38-27b-q4-local".into(),
                // already on disk in LM Studio; offload benchmark fixture
                model: "Qwen3.8-27B-Q4_K_M.gguf".into(),
                alias: "qwen3.8-27b".into(),
                ctx: 16384,
                ..Default::default()
            },
            Profile {
                name: "qwen38-27b-rewrite".into(),
                model: "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-IQ3_XXS.gguf".into(),
                alias: "qwen3.8-27b".into(),
                spec: "ngram-simple".into(),
                spec_draft_n_max: 64,
                ..Default::default()
            },
            Profile {
                name: "qwen38-flash".into(),
                model: "qwen3.8-flash-next".into(),
                alias: "qwen3.8-flash-next".into(),
                ctx: 65536,
                ctx_min: 32768,
                spec: "none".into(),
                reasoning_format: "deepseek".into(),
                ..Default::default()
            },
            Profile {
                name: "gemma4-12b".into(),
                model: "gemma-4-12B-it-Q4_K_M.gguf".into(),
                alias: "google/gemma-4-12b".into(),
                ctx: 32768,
                ctx_min: 16384,
                spec: "none".into(),
                reasoning_format: "none".into(),
                ..Default::default()
            },
            Profile {
                name: "lmstudio".into(),
                model: "qwen3.8-flash-next".into(),
                alias: "qwen3.8-flash-next".into(),
                provider: Some(EngineProvider::External),
                endpoint: "http://127.0.0.1:1234/v1".into(),
                ctx: 32768,
                ctx_min: 16384,
                spec: "none".into(),
                reasoning_format: "deepseek".into(),
                ..Default::default()
            },
        ]
    }

    pub fn is_external(&self, engine: &Engine) -> bool {
        self.provider == Some(EngineProvider::External)
            || engine.provider == EngineProvider::External
            || self.name == "lmstudio"
            || self.name == "ollama"
            || !self.endpoint.is_empty()
    }

    pub fn base_url(&self, engine: &Engine) -> String {
        if !self.endpoint.is_empty() {
            return self.endpoint.clone();
        }
        if (self.name == "lmstudio" || (self.provider == Some(EngineProvider::External) && self.alias.contains("lmstudio"))) && engine.external_url.is_empty() {
            return "http://127.0.0.1:1234/v1".to_string();
        }
        if (self.name == "ollama" || self.name.starts_with("ollama-") || self.provider == Some(EngineProvider::External)) && engine.external_url.is_empty() {
            return "http://127.0.0.1:11434/v1".to_string();
        }
        engine.base_url()
    }

    pub fn resolved_api_key(&self, engine: &Engine) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        engine.resolved_api_key()
    }

    pub fn request_model(&self) -> String {
        if !self.alias.is_empty() {
            self.alias.clone()
        } else if !self.model.is_empty() && !self.model.ends_with(".gguf") && !self.model.contains('/') && !self.model.contains('\\') {
            self.model.clone()
        } else {
            self.name.clone()
        }
    }

    pub fn ngl_auto(&self) -> bool { self.ngl.eq_ignore_ascii_case("auto") }
    pub fn ngl_value(&self) -> Option<u32> { self.ngl.parse().ok() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingSet {
    pub thinking: Sampling,
    pub coding: Sampling,
    pub instruct: Sampling,
}

impl Default for SamplingSet {
    fn default() -> Self {
        Self {
            thinking: Sampling { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 0.1, repeat_penalty: 1.1 },
            coding:   Sampling { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 0.1, repeat_penalty: 1.1 },
            instruct: Sampling { temperature: 0.7, top_p: 0.80, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repeat_penalty: 1.1 },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repeat_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self { SamplingSet::default().coding }
}

/// Reasoning effort per task class: xhigh | medium | low | none.
///
/// IMPORTANT (Qwen3.8 template): `xhigh` and `low` inject a sentence at the start of the
/// system prompt, so using a different value than the session's main effort for
/// explore/edit/chat breaks the prompt cache every time it switches. Keep those three equal
/// (default: medium, which injects nothing). `none` only changes the prompt tail and is safe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EffortMap {
    pub plan: String,
    pub explore: String,
    pub edit: String,
    pub chat: String,
    pub verify: String,
    pub summarize: String,
    pub commit: String,
}

impl Default for EffortMap {
    fn default() -> Self {
        Self {
            plan: "xhigh".into(),
            explore: "medium".into(),
            edit: "medium".into(),
            chat: "medium".into(),
            verify: "medium".into(),
            summarize: "none".into(),
            commit: "none".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// [[model_slot]]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSlot {
    pub role: String,
    pub enabled: bool,
    pub model: String,
    /// cpu | gpu
    pub device: String,
    pub port: u16,
    pub parallel: u32,
    pub ctx: u32,
    pub use_for: Vec<String>,
}

impl Default for ModelSlot {
    fn default() -> Self {
        Self {
            role: "secondary".into(),
            enabled: false,
            model: "gemma-4-12B-it-Q4_K_M.gguf".into(),
            device: "cpu".into(),
            port: 8090,
            parallel: 2,
            ctx: 8192,
            use_for: vec!["summarize".into(), "commit".into()],
        }
    }
}

// ---------------------------------------------------------------------------
// [context]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextCfg {
    pub compaction_threshold: f32,
    pub output_reserve_tokens: u32,
    pub repo_map_tokens: u32,
    pub repo_map_tokens_plan: u32,
    pub tool_output_max_bytes: usize,
    pub tool_output_head_lines: usize,
    pub tool_output_tail_lines: usize,
    pub keep_tail_turns: usize,
    pub summary_max_tokens: u32,
}

impl Default for ContextCfg {
    fn default() -> Self {
        Self {
            compaction_threshold: 0.95,
            output_reserve_tokens: 4096,
            repo_map_tokens: 2048,
            repo_map_tokens_plan: 4096,
            tool_output_max_bytes: 24_000,
            tool_output_head_lines: 150,
            tool_output_tail_lines: 50,
            keep_tail_turns: 2,
            summary_max_tokens: 1500,
        }
    }
}

// ---------------------------------------------------------------------------
// [codeintel]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CodeIntel {
    pub enabled: bool,
    pub index_path: String,
    pub languages: Vec<String>,
    pub exclude: Vec<String>,
    pub max_file_bytes: u64,
}

impl Default for CodeIntel {
    fn default() -> Self {
        Self {
            enabled: true,
            index_path: ".buzzcode/index.redb".into(),
            languages: ["rust","python","typescript","javascript","go","c","cpp","java","json","toml","markdown"]
                .into_iter().map(String::from).collect(),
            exclude: ["target/**","node_modules/**","dist/**","build/**","*.min.js","*.lock"]
                .into_iter().map(String::from).collect(),
            max_file_bytes: 1_500_000,
        }
    }
}

// ---------------------------------------------------------------------------
// [permissions]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Permissions {
    pub always_allow: Vec<String>,
    pub always_ask_shell_patterns: Vec<String>,
    /// Refuse writes outside the project dir unless yolo.
    pub confine_writes_to_project: bool,
}

impl Default for Permissions {
    fn default() -> Self {
        Self {
            always_allow: ["read_file","glob","grep","list_dir","outline","symbol_search","repo_map","git_status","git_diff","git_log"]
                .into_iter().map(String::from).collect(),
            always_ask_shell_patterns: [
                r"\brm\s+-r", r"Remove-Item\b.*-Recurse", r"git\s+push\b.*--force", r"git\s+reset\s+--hard",
                r"\bformat\b", r"rmdir\s+/s", r"del\s+/s",
            ].into_iter().map(String::from).collect(),
            confine_writes_to_project: true,
        }
    }
}

// ---------------------------------------------------------------------------
// [[mcp_server]]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpServer {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: IndexMap<String, String>,
    /// readonly | ask | yolo
    pub trust: String,
    pub enabled: bool,
    pub timeout_s: u64,
}

impl Default for McpServer {
    fn default() -> Self {
        Self { name: String::new(), command: String::new(), args: vec![], env: IndexMap::new(), trust: "ask".into(), enabled: true, timeout_s: 60 }
    }
}

// ---------------------------------------------------------------------------
// [bench] / [tui]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Bench { pub eval_dir: String }
impl Default for Bench { fn default() -> Self { Self { eval_dir: "eval/tasks".into() } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Tui {
    pub show_reasoning: bool,
    pub render_interval_ms: u64,
    pub max_transcript_blocks: usize,
    /// Show the text "arcade cabinet" side panel at startup (F3 / /factory toggles).
    pub show_factory: bool,
    /// Port for the graphical BUZZCODE ARCADE page (0 = disabled). Uses the next free port if taken.
    pub arcade_port: u16,
    /// Open the arcade page in the browser when the TUI starts.
    pub arcade_auto_open: bool,
}
impl Default for Tui {
    fn default() -> Self { Self { show_reasoning: true, render_interval_ms: 33, max_transcript_blocks: 2000, show_factory: true, arcade_port: 8123, arcade_auto_open: false } }
}

// ---------------------------------------------------------------------------

impl Engine {
    pub fn models_dir_path(&self) -> PathBuf { crate::expand_path(&self.models_dir) }
    pub fn extra_search_paths(&self) -> Vec<PathBuf> { self.extra_search_dirs.iter().map(|s| crate::expand_path(s)).collect() }
    pub fn base_url(&self) -> String {
        if self.provider == EngineProvider::External && !self.external_url.is_empty() {
            self.external_url.trim_end_matches('/').to_string()
        } else {
            format!("http://{}:{}", self.host, self.port)
        }
    }

    pub fn resolved_api_key(&self) -> Option<String> {
        if !self.api_key.trim().is_empty() {
            Some(self.api_key.trim().to_string())
        } else {
            std::env::var("BUZZCODE_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .ok()
                .filter(|s| !s.trim().is_empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roundtrips() {
        let s = toml::to_string_pretty(&Config::default()).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(back.general.default_profile, "qwen38-27b-fast");
        assert_eq!(back.profiles.len(), 7);
    }
}
