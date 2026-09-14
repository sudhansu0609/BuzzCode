//! Model catalog + config advisor: rank local (and known downloadable) GGUFs against the
//! machine's free VRAM and recommend the best quality model that still runs fast.

use crate::gguf::{GgufFile, ModelFacts};
use crate::manager::EngineManager;
use crate::nvidia::GpuInfo;
use crate::vram::{estimate_decode_tps, VramPlan};
use crate::{discover, nvidia};
use cb_config::{Config, Profile};
use serde::Serialize;
use std::path::PathBuf;

const GIB: f64 = (1u64 << 30) as f64;
/// GPU memory bandwidth used for speed estimates (GB/s). RTX 5060 Ti ≈ 448.
const GPU_BW: f64 = 448.0;
const CPU_BW: f64 = 60.0;

/// Known downloadable quants (verified file list, sizes in GiB) for the default model family.
pub const KNOWN_DOWNLOADS: &[(&str, &str, f64)] = &[
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ2_XXS.gguf", 6.77),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ2_S.gguf", 7.80),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q2_K_XL.gguf", 9.15),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ3_XXS.gguf", 10.18),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ3_S.gguf", 11.21),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q3_K_XL.gguf", 12.24),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf", 13.27),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_S.gguf", 14.30),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf", 15.33),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf", 16.35),
    ("lmstudio-community/gemma-4-12B-it-GGUF", "gemma-4-12B-it-Q4_K_M.gguf", 6.87),
    ("unsloth/Qwen3.8-Flash-Next-GGUF", "Qwen3.8-Flash-Next-Q4_K_M.gguf", 68.0),
];

#[derive(Debug, Clone, Serialize)]
pub enum Source { Local(PathBuf), Download { repo: String, file: String } }

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub name: String,
    pub source: Source,
    pub size_bytes: u64,
    pub arch: String,
    pub quant: String,
    pub params_b: f64,
    /// bits per weight (quality proxy)
    pub bpw: f64,
    pub has_mtp: bool,
    pub plan: Option<VramPlan>,
    pub fits_gpu: bool,
    pub est_tps: f64,
    pub score: f64,
    pub note: String,
    /// Existing profile name if this file is already configured.
    pub profile: Option<String>,
}

impl Candidate {
    pub fn size_gib(&self) -> f64 { self.size_bytes as f64 / GIB }
    pub fn is_local(&self) -> bool { matches!(self.source, Source::Local(_)) }
    pub fn model_spec(&self) -> String {
        match &self.source { Source::Local(p) => p.to_string_lossy().into_owned(), Source::Download { repo, file } => format!("{repo}:{file}") }
    }
}

fn base_profile(name: &str, spec: &str, has_mtp: bool) -> Profile {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let is_gemma = stem.to_ascii_lowercase().contains("gemma");
    let is_flash = stem.to_ascii_lowercase().contains("flash") || stem.to_ascii_lowercase().contains("qwen4");
    Profile {
        name: stem.to_ascii_lowercase().replace(['.', ' ', ':'], "-"),
        model: spec.to_string(),
        alias: stem.to_string(),
        spec: if has_mtp { "draft-mtp".into() } else { "none".into() },
        reasoning_format: if is_gemma { "none".into() } else { "deepseek".into() },
        ctx: if is_flash { 65536 } else if is_gemma { 32768 } else { 65536 },
        ctx_min: if is_flash { 32768 } else if is_gemma { 16384 } else { 32768 },
        ..Profile::default()
    }
}

fn evaluate(cfg: &Config, gpu: &GpuInfo, name: &str, source: Source, size_bytes: u64, facts: Option<&ModelFacts>, profile: &Profile) -> Candidate {
    let (arch, quant, params_b, has_mtp, plan) = match facts {
        Some(f) => {
            let params: u64 = estimate_params(f);
            let plan = EngineManager::plan_for(cfg, profile, f, gpu.mem.free_bytes(), gpu.mem.total_bytes());
            (f.arch.clone(), f.quant.clone(), params as f64 / 1e9, f.has_mtp, Some(plan))
        }
        None => ("?".into(), "?".into(), 0.0, false, None),
    };
    let bpw = if params_b > 0.0 { size_bytes as f64 * 8.0 / (params_b * 1e9) } else { 0.0 };
    let (fits_gpu, est_tps, note) = match (&plan, facts) {
        (Some(p), Some(f)) => {
            let fits = p.ffn_cpu_blocks == 0 && (p.ngl as usize) > f.layer_bytes.len();
            let gpu_bytes = f.weights_bytes_on_gpu(p.ngl).saturating_sub(f.ffn_tail_bytes(p.ffn_cpu_blocks as usize));
            let mtp_gain = if has_mtp && profile.spec == "draft-mtp" { 1.45 } else { 1.0 };
            let tps = estimate_decode_tps(gpu_bytes, p.est_cpu_weight_bytes, GPU_BW, CPU_BW, mtp_gain);
            let note = if fits { format!("all on GPU @ {}K ctx", p.n_ctx / 1024) }
                else if p.ffn_cpu_blocks > 0 { format!("FFN of {} blocks on CPU @ {}K ctx", p.ffn_cpu_blocks, p.n_ctx / 1024) }
                else { format!("ngl {}/{} (CPU offload) @ {}K ctx", p.ngl.min(f.n_layer), f.n_layer, p.n_ctx / 1024) };
            (fits, tps, note)
        }
        _ => (false, 0.0, "not inspected".into()),
    };
    // Quality proxy: parameter count × quantization factor (IQ2-class quants lose a lot).
    let qfactor = if bpw >= 4.0 { 1.0 } else if bpw >= 3.0 { 0.9 } else if bpw >= 2.5 { 0.7 } else { 0.5 };
    let quality = params_b * qfactor;
    let ctx_penalty = plan.as_ref().map(|p| if p.n_ctx < 32768 { 20.0 } else { 0.0 }).unwrap_or(0.0);
    // Score: fully-on-GPU models that stay ≥ 28 tok/s win; among them prefer quality.
    // Otherwise prefer speed. A slow-but-fits model still beats an offloaded one.
    let score = if fits_gpu && est_tps >= 28.0 { 1000.0 + quality * 10.0 + est_tps * 0.3 - ctx_penalty }
        else if fits_gpu { 500.0 + quality * 10.0 + est_tps - ctx_penalty }
        else { quality * 5.0 + est_tps * 3.0 };
    let profile_name = cfg.profiles.iter().find(|p| {
        match &source { Source::Local(path) => discover::find_model(cfg, &p.model).map(|m| m == *path).unwrap_or(false), Source::Download { repo, file } => p.model == format!("{repo}:{file}") }
    }).map(|p| p.name.clone());
    Candidate { name: name.to_string(), source, size_bytes, arch, quant, params_b, bpw, has_mtp, plan, fits_gpu, est_tps, score, note, profile: profile_name }
}

fn estimate_params(f: &ModelFacts) -> u64 { f.active_params() }

/// Synthetic facts for a downloadable quant: scale a local reference's tensor sizes by file size.
#[allow(dead_code)]
fn scaled_facts(reference: &ModelFacts, size_bytes: u64, quant_label: &str) -> ModelFacts {
    let ratio = size_bytes as f64 / reference.total_bytes.max(1) as f64;
    let mut f = reference.clone();
    f.layer_bytes = f.layer_bytes.iter().map(|b| (*b as f64 * ratio) as u64).collect();
    f.layer_ffn_bytes = f.layer_ffn_bytes.iter().map(|b| (*b as f64 * ratio) as u64).collect();
    f.nonlayer_bytes = (f.nonlayer_bytes as f64 * ratio) as u64;
    f.total_bytes = size_bytes;
    f.file_size = size_bytes;
    f.quant = quant_label.to_string();
    f.path = PathBuf::from(quant_label);
    f
}

#[allow(dead_code)]
fn quant_from_name(file: &str) -> String {
    let stem = file.trim_end_matches(".gguf");
    stem.rsplit("-UD-").next().map(str::to_string).unwrap_or_else(|| stem.rsplit('-').next().unwrap_or(stem).to_string())
}

/// Build the ranked candidate list of available downloaded models (from LM Studio, Ollama, and models_dir).
pub fn recommend(cfg: &Config) -> anyhow::Result<(GpuInfo, Vec<Candidate>)> {
    let gpu = nvidia::query()?;
    let mut out: Vec<Candidate> = Vec::new();

    for m in discover::discover_all_available_models(cfg) {
        let fl = m.display_name.to_ascii_lowercase();
        // Ignore multimodal projectors, standalone MTP draft modules, and non-base shards
        if fl.contains("mmproj") || fl.starts_with("mtp-") || fl.contains("-mtp-") { continue; }
        if m.display_name.contains("-of-") && !m.display_name.contains("-00001-of-") { continue; }

        let facts = GgufFile::open(&discover::first_shard(&m.path)).ok().map(|g| g.facts());
        // If file has no layer weights, it is an adapter or draft-only tensor file, not a standalone model
        if facts.as_ref().map(|f| f.layer_bytes.is_empty() || f.layer_bytes.iter().all(|&b| b == 0)).unwrap_or(false) {
            continue;
        }

        let mut prof = base_profile(&m.display_name, &m.path.to_string_lossy(), facts.as_ref().map(|f| f.has_mtp).unwrap_or(false));
        prof.alias = m.alias.clone();
        if let Some(p) = cfg.profiles.iter().find(|p| {
            discover::find_model(cfg, &p.model).map(|path| path == m.path).unwrap_or(false)
                || p.alias.eq_ignore_ascii_case(&m.alias)
                || p.model.eq_ignore_ascii_case(&m.display_name)
        }) {
            prof = p.clone();
        }

        let mut cand = evaluate(cfg, &gpu, &m.display_name, Source::Local(m.path.clone()), m.size, facts.as_ref(), &prof);
        cand.note = format!("{} · {}", cand.note, m.origin);
        out.push(cand);
    }

    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok((gpu, out))
}

/// Profile to use for a candidate (existing configured profile if any, else an ad-hoc one).
pub fn profile_for(cfg: &Config, c: &Candidate) -> Profile {
    if let Some(name) = &c.profile { if let Some(p) = cfg.profile(name) { return p.clone(); } }
    let mut p = base_profile(&c.name, &c.model_spec(), c.has_mtp);
    p.alias = c.name.clone();
    if let Some(plan) = &c.plan { p.ctx = plan.n_ctx.max(p.ctx_min); }
    p
}

/// Resolve a user selection: 1-based index, profile name, model name, alias, or path.
pub fn select<'a>(cands: &'a [Candidate], cfg: &Config, sel: &str) -> Option<&'a Candidate> {
    let sel = sel.trim();
    if let Ok(i) = sel.parse::<usize>() { return cands.get(i.wrapping_sub(1)); }
    if let Some(p) = cfg.profile(sel) {
        if let Some(c) = cands.iter().find(|c| c.profile.as_deref() == Some(&p.name) || c.name.eq_ignore_ascii_case(&p.alias) || c.name.eq_ignore_ascii_case(&p.model)) {
            return Some(c);
        }
    }
    let l = sel.to_ascii_lowercase();
    // 1. Exact match on candidate name, model_spec, or alias
    if let Some(c) = cands.iter().find(|c| c.name.eq_ignore_ascii_case(sel) || c.model_spec().eq_ignore_ascii_case(sel)) {
        return Some(c);
    }
    // 2. Tag / stem match (e.g. "qwen3.5" matches "qwen3.5:9b", "gemma-4-12b" matches "gemma-4-12B-it-Q4_K_M.gguf")
    if let Some(c) = cands.iter().find(|c| {
        c.name.split(':').next().map(|prefix| prefix.eq_ignore_ascii_case(sel)).unwrap_or(false)
            || c.name.strip_suffix(".gguf").map(|s| s.eq_ignore_ascii_case(sel)).unwrap_or(false)
    }) {
        return Some(c);
    }
    // 3. Substring match
    cands.iter().find(|c| {
        c.name.to_ascii_lowercase().contains(&l)
            || c.profile.as_deref().map(|p| p.to_ascii_lowercase().contains(&l)).unwrap_or(false)
            || c.model_spec().to_ascii_lowercase().contains(&l)
    })
}

/// Render the table (plain text, used by CLI and TUI).
pub fn render_table(gpu: &GpuInfo, cands: &[Candidate], current: Option<&str>) -> String {
    let mut s = format!("GPU {} · {} MiB free of {} MiB (other apps use {} MiB)\n", gpu.name, gpu.mem.free_mib, gpu.mem.total_mib, gpu.mem.used_mib);
    s.push_str(&format!("{:>3} {:<44} {:>7} {:>6} {:>5} {:>9} {:>4}  {}\n", "#", "model", "params", "size", "bpw", "est tok/s", "mtp", "fit / note"));
    for (i, c) in cands.iter().enumerate() {
        let star = if i == 0 { "★" } else { " " };
        let cur = if current.map(|n| c.profile.as_deref() == Some(n) || c.name.eq_ignore_ascii_case(n) || c.model_spec().ends_with(n)).unwrap_or(false) { " ◀ current" } else { "" };
        s.push_str(&format!("{star}{:>2} {:<44} {:>6.1}B {:>5.1}G {:>5.1} {:>9.0} {:>4}  {}{} {}\n",
            i + 1, truncate(&c.name, 44), c.params_b, c.size_gib(), c.bpw, c.est_tps, if c.has_mtp { "yes" } else { "no" },
            if c.fits_gpu { "✓ " } else { "⚠ " }, c.note, cur));
    }
    s.push_str("est tok/s = bandwidth model (≈ measured for Qwen3.8); KV for non-hybrid models is estimated conservatively.\n");
    if let Some(best) = cands.first() {
        s.push_str(&format!("\n★ recommended: {} — {} at ~{:.0} tok/s\n", best.name, best.note, best.est_tps));
    }
    s
}

fn truncate(s: &str, n: usize) -> String { if s.len() <= n { s.to_string() } else { format!("{}…", &s[..n - 1]) } }

