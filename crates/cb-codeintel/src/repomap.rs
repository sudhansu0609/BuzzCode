//! Ranked repository map rendered within a token budget (aider-style).

use crate::graph::SymbolGraph;
use crate::index::Indexer;
use crate::parse::SymKind;
use std::collections::HashMap;

pub struct RepoMap {
    pub graph: SymbolGraph,
}

pub struct MapOptions<'a> {
    pub budget_tokens: u32,
    /// Files the conversation is focused on (rel paths) — get extra teleport mass.
    pub focus: &'a [String],
    /// Identifiers mentioned in chat — files defining them get extra mass.
    pub mentions: &'a [String],
    /// Heuristic token estimator.
    pub estimate: &'a dyn Fn(&str) -> u32,
}

impl RepoMap {
    pub fn build(idx: &Indexer) -> Self { Self { graph: SymbolGraph::build(idx) } }

    pub fn render(&self, o: &MapOptions) -> String {
        let g = &self.graph;
        if g.files.is_empty() { return String::new(); }
        let mut pers: HashMap<usize, f32> = HashMap::new();
        for f in o.focus { if let Some(i) = g.idx_of.get(f) { *pers.entry(*i).or_default() += 10.0; } }
        for m in o.mentions {
            for (i, r) in g.recs.iter().enumerate() {
                if r.symbols.iter().any(|s| s.name.as_str() == m) { *pers.entry(i).or_default() += 5.0; }
            }
        }
        let ranks = g.pagerank(&pers, 0.85, 30);
        let mut order: Vec<usize> = (0..g.files.len()).collect();
        order.sort_by(|a, b| ranks[*b].partial_cmp(&ranks[*a]).unwrap_or(std::cmp::Ordering::Equal));

        // Binary-search the per-file symbol cap so the output fits the budget.
        let mut lo = 1usize; let mut hi = 24usize; let mut best = String::new();
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let s = self.render_with(&order, &ranks, mid, o);
            if (o.estimate)(&s) <= o.budget_tokens { best = s; lo = mid + 1; } else { hi = mid - 1; }
        }
        if best.is_empty() { best = self.render_with(&order, &ranks, 1, o); }
        best
    }

    fn render_with(&self, order: &[usize], ranks: &[f32], per_file: usize, o: &MapOptions) -> String {
        let g = &self.graph;
        let mut out = String::new();
        let mut used = 0u32;
        let total_rank: f32 = ranks.iter().sum::<f32>().max(1e-6);
        for &i in order {
            let rec = &g.recs[i];
            if rec.symbols.is_empty() { continue; }
            // Rank symbols within the file by inbound references, then publicness, then line.
            let mut syms: Vec<&crate::parse::Symbol> = rec.symbols.iter().filter(|s| s.parent.is_none() || matches!(s.kind, SymKind::Method | SymKind::Fn)).collect();
            syms.sort_by(|a, b| {
                let ia = g.symbol_inbound.get(&(i, a.name.clone())).copied().unwrap_or(0.0) + if a.is_public { 0.1 } else { 0.0 };
                let ib = g.symbol_inbound.get(&(i, b.name.clone())).copied().unwrap_or(0.0) + if b.is_public { 0.1 } else { 0.0 };
                ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal).then(a.line_start.cmp(&b.line_start))
            });
            let shown: Vec<&crate::parse::Symbol> = syms.into_iter().take(per_file).collect();
            let mut block = format!("{}:\n", g.files[i]);
            let mut shown_sorted = shown.clone();
            shown_sorted.sort_by_key(|s| s.line_start);
            for s in shown_sorted { block.push_str(&format!("│ {} {}\n", s.line_start, s.signature)); }
            let more = rec.symbols.len().saturating_sub(shown.len());
            if more > 0 { block.push_str(&format!("│ … {more} more\n")); }
            let cost = (o.estimate)(&block);
            if used + cost > o.budget_tokens { break; }
            // Stop adding files once their rank share is negligible and we have a decent map.
            if ranks[i] / total_rank < 0.002 && used > o.budget_tokens / 2 { break; }
            out.push_str(&block);
            used += cost;
        }
        out
    }
}
