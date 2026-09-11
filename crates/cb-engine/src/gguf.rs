//! Minimal GGUF v2/v3 header reader: metadata KVs + tensor infos. We never read tensor data.
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool),
    Str(String), Array(Vec<MetaValue>), U64(u64), I64(i64), F64(f64),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64, MetaValue::I8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64, MetaValue::I16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64, MetaValue::I32(v) => *v as u64,
            MetaValue::U64(v) => *v, MetaValue::I64(v) => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { MetaValue::F32(v) => Some(*v as f64), MetaValue::F64(v) => Some(*v), other => other.as_u64().map(|v| v as f64) }
    }
    pub fn as_str(&self) -> Option<&str> { if let MetaValue::Str(s) = self { Some(s) } else { None } }
    pub fn as_array(&self) -> Option<&[MetaValue]> { if let MetaValue::Array(a) = self { Some(a) } else { None } }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 { self.dims.iter().product() }
    /// Byte size on disk (== bytes in VRAM for weights), from the ggml type's block size.
    pub fn n_bytes(&self) -> u64 {
        let (block_size, type_size) = ggml_type_traits(self.ggml_type);
        let n = self.n_elements();
        n.div_ceil(block_size) * type_size
    }
    pub fn block_index(&self) -> Option<u32> {
        let rest = self.name.strip_prefix("blk.")?;
        let end = rest.find('.')?;
        rest[..end].parse().ok()
    }
}

/// (block_size, type_size_bytes) for ggml types. Mirrors ggml.c's type_traits.
pub fn ggml_type_traits(t: u32) -> (u64, u64) {
    match t {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        _ => (1, 2),      // unknown: assume 16-bit
    }
}

pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K",
        16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S",
        22 => "IQ2_S", 23 => "IQ4_XS", 24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64", 28 => "F64",
        29 => "IQ1_M", 30 => "BF16", 34 => "TQ1_0", 35 => "TQ2_0", 39 => "MXFP4", _ => "?",
    }
}

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub file_size: u64,
}

struct Reader<R: Read> { r: R }

impl<R: Read> Reader<R> {
    fn u8(&mut self) -> Result<u8> { let mut b = [0u8; 1]; self.r.read_exact(&mut b)?; Ok(b[0]) }
    fn u16(&mut self) -> Result<u16> { let mut b = [0u8; 2]; self.r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
    fn u32(&mut self) -> Result<u32> { let mut b = [0u8; 4]; self.r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
    fn u64(&mut self) -> Result<u64> { let mut b = [0u8; 8]; self.r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
    fn f32(&mut self) -> Result<f32> { Ok(f32::from_bits(self.u32()?)) }
    fn f64(&mut self) -> Result<f64> { Ok(f64::from_bits(self.u64()?)) }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        if len > 64 * 1024 * 1024 { bail!("gguf string too long ({len})"); }
        let mut buf = vec![0u8; len];
        self.r.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<MetaValue> {
        Ok(match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(self.f32()?),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::Str(self.string()?),
            9 => {
                if depth > 4 { bail!("gguf array nesting too deep"); }
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                if n > 50_000_000 { bail!("gguf array too long ({n})"); }
                // Large token arrays: keep them but as cheaply as possible.
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n { v.push(self.value(ety, depth + 1)?); }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(self.f64()?),
            other => bail!("unknown gguf value type {other}"),
        })
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_size = file.metadata()?.len();
        let mut rd = Reader { r: BufReader::with_capacity(1 << 20, file) };
        let magic = rd.u32()?;
        if magic != GGUF_MAGIC { bail!("{} is not a GGUF file (magic {magic:#x})", path.display()); }
        let version = rd.u32()?;
        if !(2..=3).contains(&version) { bail!("unsupported GGUF version {version}"); }
        let n_tensors = rd.u64()? as usize;
        let n_kv = rd.u64()? as usize;
        if n_tensors > 1_000_000 || n_kv > 1_000_000 { bail!("implausible GGUF header counts"); }

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = rd.string()?;
            let ty = rd.u32()?;
            let val = rd.value(ty, 0)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = rd.string()?;
            let n_dims = rd.u32()? as usize;
            if n_dims > 8 { bail!("tensor {name} has {n_dims} dims"); }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims { dims.push(rd.u64()?); }
            let ggml_type = rd.u32()?;
            let offset = rd.u64()?;
            tensors.push(TensorInfo { name, dims, ggml_type, offset });
        }
        // Split files: only the first shard has the full tensor list for its part; callers pass shard 1.
        let _ = rd.r.seek(SeekFrom::Current(0));
        Ok(Self { path: path.to_path_buf(), version, metadata, tensors, file_size })
    }

    pub fn arch(&self) -> String {
        self.metadata.get("general.architecture").and_then(MetaValue::as_str).unwrap_or("unknown").to_string()
    }

    fn arch_key(&self, suffix: &str) -> Option<&MetaValue> {
        self.metadata.get(&format!("{}.{}", self.arch(), suffix))
    }

    fn arch_u64(&self, suffix: &str) -> Option<u64> { self.arch_key(suffix).and_then(MetaValue::as_u64) }

    pub fn facts(&self) -> ModelFacts {
        let arch = self.arch();
        // Qwen3.5/3.8 GGUFs count the MTP ("nextn") block(s) in block_count; they are not part of
        // the main forward pass, so we split them out for planning.
        let n_nextn = self.arch_u64("nextn_predict_layers").unwrap_or(0) as u32;
        let n_layer_total = self.arch_u64("block_count").unwrap_or(0) as u32;
        let n_layer = n_layer_total.saturating_sub(n_nextn);
        let n_embd = self.arch_u64("embedding_length").unwrap_or(0) as u32;
        let n_head = self.arch_u64("attention.head_count").unwrap_or(0) as u32;
        // head_count_kv may be an array (per-layer) for hybrid models: 0 = recurrent layer.
        let (n_head_kv, attn_layers): (u32, Vec<bool>) = match self.arch_key("attention.head_count_kv") {
            Some(MetaValue::Array(a)) => {
                let per: Vec<u64> = a.iter().filter_map(MetaValue::as_u64).collect();
                let kv = per.iter().copied().filter(|&v| v > 0).max().unwrap_or(0) as u32;
                (kv, per.iter().map(|&v| v > 0).collect())
            }
            Some(v) => (v.as_u64().unwrap_or(0) as u32, vec![]),
            None => (0, vec![]),
        };
        let n_head_kv = if n_head_kv == 0 { n_head } else { n_head_kv };
        let head_dim = self.arch_u64("attention.key_length")
            .or_else(|| if n_head > 0 && n_embd > 0 { Some((n_embd / n_head) as u64) } else { None })
            .unwrap_or(128) as u32;
        let value_dim = self.arch_u64("attention.value_length").unwrap_or(head_dim as u64) as u32;
        let n_ctx_train = self.arch_u64("context_length").unwrap_or(32768) as u32;

        // Determine attention vs recurrent layers from tensor names when metadata lacks the array.
        let mut is_attn = vec![false; n_layer as usize];
        let mut is_recurrent = vec![false; n_layer as usize];
        let mut layer_bytes = vec![0u64; n_layer as usize];
        let mut layer_ffn_bytes = vec![0u64; n_layer as usize];
        let mut nonlayer_bytes = 0u64;
        let mut has_mtp = false;
        let mut mtp_bytes = 0u64;
        let mut ssm_state_bytes = 0u64; // per-seq recurrent state (conv + delta state) approximation
        let mut n_params = 0u64;
        for t in &self.tensors {
            let bytes = t.n_bytes();
            if t.name.contains("nextn") || t.name.contains("mtp") {
                has_mtp = true;
                mtp_bytes += bytes;
                continue;
            }
            if !(t.block_index().map(|b| b >= n_layer && b < n_layer_total).unwrap_or(false)) { n_params += t.n_elements(); }
            match t.block_index() {
                Some(b) if b >= n_layer && b < n_layer_total => {
                    // trailing nextn/MTP block
                    has_mtp = true;
                    mtp_bytes += bytes;
                }
                Some(b) if (b as usize) < layer_bytes.len() => {
                    let b = b as usize;
                    layer_bytes[b] += bytes;
                    if t.name.contains(".ffn_") { layer_ffn_bytes[b] += bytes; }
                    if t.name.contains(".attn_k.") || t.name.contains(".attn_v.") || t.name.contains(".attn_kv.") {
                        is_attn[b] = true;
                    }
                    if t.name.contains(".ssm_") || t.name.contains(".attn_gate") || t.name.contains(".linear_attn") {
                        is_recurrent[b] = true;
                    }
                }
                _ => nonlayer_bytes += bytes,
            }
        }
        if !attn_layers.is_empty() && attn_layers.len() >= n_layer as usize {
            for (i, a) in attn_layers.iter().take(n_layer as usize).enumerate() { is_attn[i] = *a; if !*a { is_recurrent[i] = true; } }
        }
        // Hybrid models without per-layer metadata but with a full_attention_interval: every k-th layer is attention.
        if attn_layers.is_empty() {
            if let Some(k) = self.arch_u64("full_attention_interval").filter(|&k| k > 1) {
                for i in 0..n_layer as usize {
                    let attn = (i as u64 + 1) % k == 0;
                    is_attn[i] = attn;
                    is_recurrent[i] = !attn;
                }
            }
        }
        if n_layer > 0 && is_recurrent.iter().all(|&r| !r) && is_attn.iter().all(|&a| !a) {
            is_attn.fill(true);
        }
        let n_attn_layers = is_attn.iter().filter(|&&a| a).count() as u32;
        let n_recurrent_layers = is_recurrent.iter().filter(|&&r| r).count() as u32;

        // Recurrent state estimate (Gated DeltaNet / Mamba2-style): per layer
        //   conv state: d_conv * d_inner ; delta state: n_heads * head_dim_k * head_dim_v
        // Keys vary by arch; fall back to a conservative constant per layer if absent.
        if n_recurrent_layers > 0 {
            let d_conv = self.arch_u64("ssm.conv_kernel").unwrap_or(4);
            let d_inner = self.arch_u64("ssm.inner_size").unwrap_or(n_embd as u64 * 2);
            let d_state = self.arch_u64("ssm.state_size").unwrap_or(128);
            let n_ssm_head = self.arch_u64("ssm.time_step_rank").or_else(|| self.arch_u64("ssm.group_count")).unwrap_or(32);
            let per_layer = (d_conv * d_inner + n_ssm_head * d_state * d_state) * 4; // f32 state
            ssm_state_bytes = per_layer * n_recurrent_layers as u64;
        }

        let n_vocab = self.metadata.get("tokenizer.ggml.tokens").and_then(MetaValue::as_array).map(|a| a.len()).unwrap_or(0) as u32;
        let chat_template = self.metadata.get("tokenizer.chat_template").and_then(MetaValue::as_str).map(str::to_string);
        let name = self.metadata.get("general.name").and_then(MetaValue::as_str).map(str::to_string).unwrap_or_default();
        let quant = self.metadata.get("general.file_type").and_then(MetaValue::as_u64).map(file_type_name).unwrap_or("?").to_string();
        let total_bytes: u64 = self.tensors.iter().map(TensorInfo::n_bytes).sum();
        let expert_count = self.arch_u64("expert_count").unwrap_or(0) as u32;
        let expert_used_count = self.arch_u64("expert_used_count").unwrap_or(0) as u32;

        ModelFacts {
            path: self.path.clone(), name, arch, quant, n_layer, n_embd, n_head, n_head_kv, head_dim, value_dim,
            n_attn_layers, n_recurrent_layers, is_attn, n_ctx_train, n_vocab, has_mtp, mtp_bytes,
            layer_bytes, layer_ffn_bytes, nonlayer_bytes, total_bytes, file_size: self.file_size,
            recurrent_state_bytes_per_seq: ssm_state_bytes, chat_template, n_params,
            expert_count, expert_used_count,
        }
    }
}

fn file_type_name(ft: u64) -> &'static str {
    match ft {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 7 => "Q8_0", 8 => "Q5_0", 9 => "Q5_1", 10 => "Q2_K",
        11 => "Q3_K_S", 12 => "Q3_K_M", 13 => "Q3_K_L", 14 => "Q4_K_S", 15 => "Q4_K_M", 16 => "Q5_K_S", 17 => "Q5_K_M",
        18 => "Q6_K", 19 => "IQ2_XXS", 20 => "IQ2_XS", 21 => "Q2_K_S", 22 => "IQ3_XS", 23 => "IQ3_XXS", 24 => "IQ1_S",
        25 => "IQ4_NL", 26 => "IQ3_S", 27 => "IQ3_M", 28 => "IQ2_S", 29 => "IQ2_M", 30 => "IQ4_XS", 31 => "IQ1_M",
        32 => "BF16", 36 => "TQ1_0", 37 => "TQ2_0", 38 => "MXFP4", _ => "mixed",
    }
}

/// Everything the planner and the harness need to know about a model, derived from the GGUF header.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelFacts {
    pub path: PathBuf,
    pub name: String,
    pub arch: String,
    pub quant: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub value_dim: u32,
    pub n_attn_layers: u32,
    pub n_recurrent_layers: u32,
    pub is_attn: Vec<bool>,
    pub n_ctx_train: u32,
    pub n_vocab: u32,
    pub has_mtp: bool,
    pub mtp_bytes: u64,
    pub layer_bytes: Vec<u64>,
    pub layer_ffn_bytes: Vec<u64>,
    pub nonlayer_bytes: u64,
    pub total_bytes: u64,
    pub file_size: u64,
    pub recurrent_state_bytes_per_seq: u64,
    #[serde(skip)]
    pub chat_template: Option<String>,
    /// Exact parameter count (tensor elements, excluding the MTP head).
    pub n_params: u64,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

impl ModelFacts {
    pub fn is_hybrid(&self) -> bool { self.n_recurrent_layers > 0 && self.n_attn_layers > 0 }
    pub fn is_moe(&self) -> bool { self.expert_count > 0 && self.expert_used_count > 0 }
    pub fn active_params(&self) -> u64 {
        if self.is_moe() {
            let ratio = self.expert_used_count as f64 / self.expert_count as f64;
            (self.n_params as f64 * ratio.clamp(0.02, 1.0)) as u64
        } else {
            self.n_params
        }
    }
    pub fn weights_bytes_on_gpu(&self, ngl: u32) -> u64 {
        let n = (ngl as usize).min(self.layer_bytes.len());
        let layers: u64 = self.layer_bytes[..n].iter().sum();
        // llama.cpp puts output/embeddings on GPU once ngl > n_layer
        let non = if ngl as usize > self.layer_bytes.len() { self.nonlayer_bytes } else { 0 };
        layers + non
    }
    /// Bytes moved to CPU when the FFN tensors of the last `k` blocks are overridden.
    pub fn ffn_tail_bytes(&self, k: usize) -> u64 {
        let n = self.layer_ffn_bytes.len();
        let k = k.min(n);
        self.layer_ffn_bytes[n - k..].iter().sum()
    }
    pub fn summary(&self) -> String {
        format!(
            "{} [{}] {} quant={} layers={} (attn {}, recurrent {}) kv_heads={} head_dim={} ctx_train={} mtp={} weights={:.2} GiB",
            self.name, self.arch, self.path.file_name().map(|s| s.to_string_lossy()).unwrap_or_default(),
            self.quant, self.n_layer, self.n_attn_layers, self.n_recurrent_layers, self.n_head_kv, self.head_dim,
            self.n_ctx_train, self.has_mtp, self.total_bytes as f64 / (1u64 << 30) as f64
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_bytes() {
        let t = TensorInfo { name: "blk.0.ffn_up.weight".into(), dims: vec![256, 2], ggml_type: 12, offset: 0 };
        assert_eq!(t.n_bytes(), 2 * 144);
        assert_eq!(t.block_index(), Some(0));
    }
}
