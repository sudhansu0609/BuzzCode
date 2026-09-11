//! File dependency graph from symbol references + personalized PageRank.

use crate::index::{FileRec, Indexer};
use smol_str::SmolStr;
use std::collections::HashMap;
use std::sync::Arc;

pub struct SymbolGraph {
    pub files: Vec<String>,
    pub idx_of: HashMap<String, usize>,
    /// Outgoing weighted edges: from file i → (file j, weight).
    pub edges: Vec<Vec<(usize, f32)>>,
    /// Inbound reference weight per (file, symbol name) — used to pick which symbols to show.
    pub symbol_inbound: HashMap<(usize, SmolStr), f32>,
    pub recs: Vec<Arc<FileRec>>,
}

impl SymbolGraph {
    pub fn build(idx: &Indexer) -> Self {
        let mut entries: Vec<(String, Arc<FileRec>)> = idx.files().into_iter().filter(|(_, r)| r.lang.is_code()).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let files: Vec<String> = entries.iter().map(|e| e.0.clone()).collect();
        let recs: Vec<Arc<FileRec>> = entries.iter().map(|e| e.1.clone()).collect();
        let idx_of: HashMap<String, usize> = files.iter().enumerate().map(|(i, f)| (f.clone(), i)).collect();

        // name → [(file, kind weight, is_public)]
        let mut defs: HashMap<&str, Vec<(usize, f32, bool)>> = HashMap::new();
        for (i, r) in recs.iter().enumerate() {
            for s in &r.symbols {
                if s.name.len() < 3 { continue; }
                defs.entry(s.name.as_str()).or_default().push((i, s.kind.weight(), s.is_public));
            }
        }
        // Names defined in too many files (e.g. "new", "main", "test") are noise.
        let common: std::collections::HashSet<&str> = defs.iter().filter(|(_, v)| v.len() > 12).map(|(k, _)| *k).collect();

        let mut edges: Vec<HashMap<usize, f32>> = vec![HashMap::new(); files.len()];
        let mut symbol_inbound: HashMap<(usize, SmolStr), f32> = HashMap::new();
        for (i, r) in recs.iter().enumerate() {
            for (name, count) in &r.refs {
                if common.contains(name.as_str()) { continue; }
                let Some(targets) = defs.get(name.as_str()) else { continue };
                let share = 1.0 / targets.len() as f32;
                for (j, kw, is_pub) in targets {
                    if *j == i { continue; }
                    let w = (*count as f32).sqrt() * kw * share * if *is_pub { 1.0 } else { 0.5 };
                    *edges[i].entry(*j).or_default() += w;
                    *symbol_inbound.entry((*j, name.clone())).or_default() += w;
                }
            }
        }
        let edges = edges.into_iter().map(|m| m.into_iter().collect()).collect();
        Self { files, idx_of, edges, symbol_inbound, recs }
    }

    /// Personalized PageRank. `personalization` = extra teleport mass per file index.
    pub fn pagerank(&self, personalization: &HashMap<usize, f32>, damping: f32, iters: usize) -> Vec<f32> {
        let n = self.files.len();
        if n == 0 { return vec![]; }
        let mut tele = vec![1.0f32; n];
        for (i, w) in personalization { if let Some(t) = tele.get_mut(*i) { *t += *w; } }
        let tsum: f32 = tele.iter().sum();
        for t in &mut tele { *t /= tsum; }
        let out_w: Vec<f32> = self.edges.iter().map(|e| e.iter().map(|(_, w)| *w).sum::<f32>()).collect();
        let mut rank = tele.clone();
        for _ in 0..iters {
            let mut next = vec![0.0f32; n];
            let mut dangling = 0.0f32;
            for i in 0..n {
                if out_w[i] <= 0.0 { dangling += rank[i]; continue; }
                for (j, w) in &self.edges[i] { next[*j] += damping * rank[i] * (*w / out_w[i]); }
            }
            for i in 0..n { next[i] += (1.0 - damping) * tele[i] + damping * dangling * tele[i]; }
            rank = next;
        }
        rank
    }
}
