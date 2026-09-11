//! VRAM budget model + planner: chooses `-ngl`, context size and `--override-tensor` offload
//! so that the model fits in *free* VRAM with a safety margin.
//!
//! Budget = weights_on_gpu + kv_cache(ctx) + recurrent_state × (1 + checkpoints)
//!        + compute_buffer(ubatch) + mtp_head + cuda_context + safety

use crate::gguf::ModelFacts;
use serde::{Deserialize, Serialize};

const MIB: u64 = 1 << 20;
const GIB: f64 = (1u64 << 30) as f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvType { F16, BF16, Q8_0, Q4_0 }

impl KvType {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "f16" => KvType::F16, "bf16" => KvType::BF16, "q4_0" => KvType::Q4_0, _ => KvType::Q8_0,
        }
    }
    /// bytes per element including block overhead
    pub fn bytes_per_elem(self) -> f64 {
        match self { KvType::F16 | KvType::BF16 => 2.0, KvType::Q8_0 => 34.0 / 32.0, KvType::Q4_0 => 18.0 / 32.0 }
    }
    pub fn flag(self) -> &'static str {
        match self { KvType::F16 => "f16", KvType::BF16 => "bf16", KvType::Q8_0 => "q8_0", KvType::Q4_0 => "q4_0" }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlanOpts {
    pub kv_type: KvType,
    pub ubatch: u32,
    pub ctx_checkpoints: u32,
    pub mtp: bool,
    pub cuda_context_bytes: u64,
    pub safety_bytes: u64,
}

impl Default for PlanOpts {
    fn default() -> Self {
        Self { kv_type: KvType::Q8_0, ubatch: 512, ctx_checkpoints: 8, mtp: true, cuda_context_bytes: 600 * MIB, safety_bytes: 350 * MIB }
    }
}

#[derive(Debug, Clone)]
pub struct PlanPrefs {
    pub pref_ctx: u32,
    pub min_ctx: u32,
    pub allow_offload: bool,
    /// Force a specific ngl (skip search).
    pub fixed_ngl: Option<u32>,
    /// Force a specific override-tensor expression ("" = none).
    pub fixed_ot: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramPlan {
    pub ngl: u32,
    pub n_ctx: u32,
    /// `-ot` expression, if any.
    pub override_tensor: Option<String>,
    /// Number of trailing blocks whose FFN lives on CPU (0 = none).
    pub ffn_cpu_blocks: u32,
    pub ctx_checkpoints: u32,
    pub est_vram_bytes: u64,
    pub est_cpu_weight_bytes: u64,
    pub free_vram_bytes: u64,
    pub rationale: String,
}

impl VramPlan {
    pub fn est_vram_gib(&self) -> f64 { self.est_vram_bytes as f64 / GIB }
}

pub struct VramPlanner<'a> {
    pub facts: &'a ModelFacts,
    pub free_vram: u64,
    pub total_vram: u64,
    pub free_ram: u64,
}

impl<'a> VramPlanner<'a> {
    pub fn kv_bytes_per_token(&self, kv: KvType) -> u64 {
        let f = self.facts;
        let per_layer = (f.n_head_kv as f64) * ((f.head_dim + f.value_dim) as f64) * kv.bytes_per_elem();
        (per_layer * f.n_attn_layers as f64).ceil() as u64
    }

    fn compute_bytes(&self, ubatch: u32) -> u64 {
        // Empirical: ~ ubatch × (n_embd × 8 + n_vocab × 4) f32-ish temporaries + attention scores.
        let f = self.facts;
        let per_tok = (f.n_embd as u64) * 4 * 8 + (f.n_vocab.max(32_000) as u64) * 4;
        (ubatch as u64 * per_tok).max(256 * MIB).min(1536 * MIB)
    }

    /// Bytes needed in VRAM for (ngl, n_ctx, ffn_cpu_blocks).
    pub fn estimate(&self, ngl: u32, n_ctx: u32, ffn_cpu_blocks: u32, o: &PlanOpts) -> u64 {
        let f = self.facts;
        let weights = f.weights_bytes_on_gpu(ngl).saturating_sub(f.ffn_tail_bytes(ffn_cpu_blocks as usize));
        let kv = self.kv_bytes_per_token(o.kv_type) * n_ctx as u64;
        // Measured on Qwen3.8-27B: checkpoints cost roughly half the naive per-checkpoint state size.
        let rec = f.recurrent_state_bytes_per_seq + f.recurrent_state_bytes_per_seq * o.ctx_checkpoints as u64 / 2;
        let mtp = if o.mtp && f.has_mtp { f.mtp_bytes + 64 * MIB } else { 0 };
        weights + kv + rec + self.compute_bytes(o.ubatch) + mtp + o.cuda_context_bytes + o.safety_bytes
    }

    pub fn plan(&self, prefs: &PlanPrefs, o: &PlanOpts) -> VramPlan {
        let f = self.facts;
        let usable = self.free_vram;
        let all_layers = f.n_layer + 1; // +1 = output layer on GPU
        let mut rationale = Vec::new();
        rationale.push(format!(
            "free VRAM {:.2} GiB; weights {:.2} GiB; kv/token {} B ({} attn layers, {}); recurrent state {:.0} MiB/seq",
            usable as f64 / GIB, f.total_bytes as f64 / GIB, self.kv_bytes_per_token(o.kv_type),
            f.n_attn_layers, o.kv_type.flag(), f.recurrent_state_bytes_per_seq as f64 / MIB as f64
        ));

        let mk = |ngl: u32, n_ctx: u32, k: u32, ot: Option<String>, est: u64, r: Vec<String>| VramPlan {
            ngl, n_ctx, override_tensor: ot, ffn_cpu_blocks: k, ctx_checkpoints: o.ctx_checkpoints,
            est_vram_bytes: est, est_cpu_weight_bytes: f.total_bytes.saturating_sub(f.mtp_bytes).saturating_sub(f.weights_bytes_on_gpu(ngl)) + f.ffn_tail_bytes(k as usize),
            free_vram_bytes: usable, rationale: r.join("\n"),
        };

        // Fixed overrides from the profile.
        if let Some(ngl) = prefs.fixed_ngl {
            let k = 0;
            let est = self.estimate(ngl, prefs.pref_ctx, k, o);
            rationale.push(format!("profile fixes ngl={ngl}; est {:.2} GiB", est as f64 / GIB));
            return mk(ngl, prefs.pref_ctx, k, prefs.fixed_ot.clone().filter(|s| !s.is_empty()), est, rationale);
        }

        // 1. Everything on GPU at preferred ctx.
        let est = self.estimate(all_layers, prefs.pref_ctx, 0, o);
        if est <= usable {
            rationale.push(format!("all {} layers on GPU @ ctx {} → {:.2} GiB (fits)", f.n_layer, prefs.pref_ctx, est as f64 / GIB));
            return mk(all_layers, prefs.pref_ctx, 0, None, est, rationale);
        }
        rationale.push(format!("all layers @ ctx {} → {:.2} GiB (too big)", prefs.pref_ctx, est as f64 / GIB));

        // 2. Reduce context.
        let mut ctx = prefs.pref_ctx;
        while ctx > prefs.min_ctx {
            ctx = (ctx - 4096).max(prefs.min_ctx);
            let est = self.estimate(all_layers, ctx, 0, o);
            if est <= usable {
                rationale.push(format!("all layers @ reduced ctx {ctx} → {:.2} GiB (fits)", est as f64 / GIB));
                return mk(all_layers, ctx, 0, None, est, rationale);
            }
        }

        if !prefs.allow_offload {
            let est = self.estimate(all_layers, prefs.min_ctx, 0, o);
            rationale.push("offload disabled; will likely OOM".into());
            return mk(all_layers, prefs.min_ctx, 0, None, est, rationale);
        }

        // 3. Offload FFN of trailing blocks (finer than -ngl; keeps attention + recurrent on GPU).
        //    Prefer keeping the *preferred* ctx if offloading a little achieves it; otherwise use min_ctx.
        let n = f.n_layer;
        for &target_ctx in &[prefs.pref_ctx, (prefs.pref_ctx + prefs.min_ctx) / 2 / 4096 * 4096, prefs.min_ctx] {
            if target_ctx < prefs.min_ctx { continue; }
            for k in 1..=n {
                let est = self.estimate(all_layers, target_ctx, k, o);
                if est <= usable {
                    // Don't offload more than ~40% of layers' FFN this way; beyond that -ngl is simpler.
                    if k * 10 <= n * 4 {
                        let ot = ffn_override_regex(n, k);
                        rationale.push(format!("FFN of last {k} blocks → CPU ({:.2} GiB), ctx {target_ctx} → {:.2} GiB (fits)",
                            f.ffn_tail_bytes(k as usize) as f64 / GIB, est as f64 / GIB));
                        return mk(all_layers, target_ctx, k, Some(ot), est, rationale);
                    }
                    break;
                }
            }
        }

        // 4. Plain -ngl fallback.
        let mut ngl = n;
        while ngl > 0 {
            let est = self.estimate(ngl, prefs.min_ctx, 0, o);
            if est <= usable {
                rationale.push(format!("ngl {ngl}/{n} @ ctx {} → {:.2} GiB (fits; expect slow decode)", prefs.min_ctx, est as f64 / GIB));
                return mk(ngl, prefs.min_ctx, 0, None, est, rationale);
            }
            ngl = ngl.saturating_sub(2);
        }
        let est = self.estimate(0, prefs.min_ctx, 0, o);
        rationale.push("model does not fit on GPU at all; CPU inference".into());
        mk(0, prefs.min_ctx, 0, None, est, rationale)
    }
}

/// Build a `-ot` regex that sends the FFN tensors of the last `k` of `n` blocks to CPU.
pub fn ffn_override_regex(n: u32, k: u32) -> String {
    let start = n.saturating_sub(k);
    let ids: Vec<String> = (start..n).map(|i| i.to_string()).collect();
    format!(r"blk\.({})\.ffn_.*=CPU", ids.join("|"))
}

/// Crude decode-speed estimate (tok/s) from memory bandwidth and bytes read per token.
pub fn estimate_decode_tps(gpu_bytes: u64, cpu_bytes: u64, gpu_bw_gbs: f64, cpu_bw_gbs: f64, mtp_gain: f64) -> f64 {
    let t_gpu = gpu_bytes as f64 / (gpu_bw_gbs * 1e9);
    let t_cpu = cpu_bytes as f64 / (cpu_bw_gbs * 1e9);
    let eff = 0.70;
    (1.0 / (t_gpu + t_cpu)) * eff * mtp_gain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_facts(layers: u32, per_layer_bytes: u64) -> ModelFacts {
        ModelFacts {
            path: "x.gguf".into(), name: "t".into(), arch: "qwen3next".into(), quant: "Q3".into(),
            n_layer: layers, n_embd: 4096, n_head: 32, n_head_kv: 4, head_dim: 128, value_dim: 128,
            n_attn_layers: layers / 4, n_recurrent_layers: layers - layers / 4, is_attn: vec![],
            n_ctx_train: 262144, n_vocab: 150_000, has_mtp: true, mtp_bytes: 200 * MIB,
            layer_bytes: vec![per_layer_bytes; layers as usize],
            layer_ffn_bytes: vec![per_layer_bytes * 7 / 10; layers as usize],
            nonlayer_bytes: 900 * MIB, total_bytes: per_layer_bytes * layers as u64 + 900 * MIB, file_size: 0,
            recurrent_state_bytes_per_seq: 100 * MIB, chat_template: None, n_params: 27_000_000_000,
            expert_count: 0, expert_used_count: 0,
        }
    }

    #[test]
    fn fits_all_on_gpu() {
        let f = fake_facts(64, 140 * MIB); // ~8.75 GiB weights + ~3 GiB overheads
        let p = VramPlanner { facts: &f, free_vram: 13_700 * MIB, total_vram: 16_311 * MIB, free_ram: 20 << 30 };
        let plan = p.plan(&PlanPrefs { pref_ctx: 32768, min_ctx: 16384, allow_offload: true, fixed_ngl: None, fixed_ot: None }, &PlanOpts::default());
        assert_eq!(plan.ngl, 65);
        assert_eq!(plan.n_ctx, 32768);
        assert!(plan.override_tensor.is_none());
    }

    #[test]
    fn offloads_when_too_big() {
        let f = fake_facts(64, 250 * MIB); // ~15.6 GiB like Q4_K_M
        let p = VramPlanner { facts: &f, free_vram: 13_700 * MIB, total_vram: 16_311 * MIB, free_ram: 20 << 30 };
        let plan = p.plan(&PlanPrefs { pref_ctx: 32768, min_ctx: 16384, allow_offload: true, fixed_ngl: None, fixed_ot: None }, &PlanOpts::default());
        assert!(plan.override_tensor.is_some() || plan.ngl < 65, "{}", plan.rationale);
        assert!(plan.est_vram_bytes <= 13_700 * MIB);
    }

    #[test]
    fn regex_shape() {
        assert_eq!(ffn_override_regex(64, 2), r"blk\.(62|63)\.ffn_.*=CPU");
    }
}
