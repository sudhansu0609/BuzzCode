//! Token counting: exact via llama-server `/tokenize` (cached by blake3), heuristic fallback,
//! and calibration from the server's reported `prompt_n + cache_n`.

use cb_engine::LlamaClient;
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct TokenCounter {
    client: Option<LlamaClient>,
    cache: Mutex<LruCache<[u8; 32], u32>>,
    /// chars-per-token × 1000 (atomic f32 substitute). Learned; starts at 3.6.
    cpt_milli: AtomicU64,
}

impl TokenCounter {
    pub fn new(client: Option<LlamaClient>) -> Self {
        Self { client, cache: Mutex::new(LruCache::new(NonZeroUsize::new(4096).unwrap())), cpt_milli: AtomicU64::new(3600) }
    }

    pub fn chars_per_token(&self) -> f32 { self.cpt_milli.load(Ordering::Relaxed) as f32 / 1000.0 }

    /// Synchronous heuristic.
    pub fn estimate(&self, text: &str) -> u32 {
        if text.is_empty() { return 0; }
        // Code/JSON tokenizes denser than prose; bias by punctuation ratio.
        let punct = text.bytes().filter(|b| b"{}[]()<>=;:,./\\\"'`".contains(b)).count() as f32 / text.len() as f32;
        let cpt = (self.chars_per_token() * (1.0 - punct * 0.6)).max(2.2);
        (text.len() as f32 / cpt).ceil() as u32
    }

    /// Exact count via server; falls back to heuristic if the server is unavailable.
    pub async fn count(&self, text: &str) -> u32 {
        if text.is_empty() { return 0; }
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(n) = self.cache.lock().get(&key).copied() { return n; }
        let n = match &self.client {
            Some(c) => match c.tokenize_count(text).await { Ok(n) => n, Err(_) => self.estimate(text) },
            None => self.estimate(text),
        };
        self.cache.lock().put(key, n);
        n
    }

    /// Update the chars/token ratio from a whole-prompt observation.
    pub fn calibrate(&self, prompt_tokens: u32, prompt_chars: usize) {
        if prompt_tokens < 200 || prompt_chars < 800 { return; }
        let observed = prompt_chars as f32 / prompt_tokens as f32;
        let cur = self.chars_per_token();
        let next = cur * 0.7 + observed * 0.3;
        self.cpt_milli.store((next.clamp(2.0, 6.0) * 1000.0) as u64, Ordering::Relaxed);
    }
}
