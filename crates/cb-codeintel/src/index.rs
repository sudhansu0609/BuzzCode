//! Persistent symbol index (redb) with an in-memory mirror for fast queries.

use crate::lang::Language;
use crate::parse::{self, SymKind, Symbol};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use rayon::prelude::*;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const FILES: TableDefinition<&str, &[u8]> = TableDefinition::new("files");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const INDEX_VERSION: &str = "3";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRec {
    pub mtime_ns: u64,
    pub size: u64,
    pub hash: [u8; 32],
    pub lang: Language,
    pub symbols: Vec<Symbol>,
    pub refs: Vec<(SmolStr, u16)>,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct ScanStats { pub files_seen: usize, pub parsed: usize, pub unchanged: usize, pub removed: usize, pub symbols: usize, pub ms: u128 }

pub struct Indexer {
    db: Database,
    root: PathBuf,
    exclude: Vec<String>,
    languages: Vec<String>,
    max_file_bytes: u64,
    /// rel path (forward slashes) → record
    files: RwLock<HashMap<String, Arc<FileRec>>>,
}

impl Indexer {
    pub fn open(root: &Path, db_path: &Path, cfg: &cb_config::CodeIntel) -> Result<Self> {
        if let Some(p) = db_path.parent() { std::fs::create_dir_all(p)?; }
        let db = Database::create(db_path).with_context(|| format!("opening index {}", db_path.display()))?;
        let me = Self {
            db, root: root.to_path_buf(), exclude: cfg.exclude.clone(), languages: cfg.languages.clone(), max_file_bytes: cfg.max_file_bytes,
            files: RwLock::new(HashMap::new()),
        };
        me.load()?;
        Ok(me)
    }

    fn load(&self) -> Result<()> {
        let txn = self.db.begin_read()?;
        let version_ok = match txn.open_table(META) {
            Ok(t) => t.get("version")?.map(|v| v.value() == INDEX_VERSION).unwrap_or(false),
            Err(_) => false,
        };
        if !version_ok {
            // Wipe on schema change.
            let w = self.db.begin_write()?;
            { let mut t = w.open_table(FILES)?; let keys: Vec<String> = t.iter()?.flatten().map(|(k, _)| k.value().to_string()).collect(); for k in keys { t.remove(k.as_str())?; } }
            { let mut m = w.open_table(META)?; m.insert("version", INDEX_VERSION)?; }
            w.commit()?;
            return Ok(());
        }
        let t = txn.open_table(FILES)?;
        let mut map = HashMap::new();
        for item in t.iter()? {
            let (k, v) = item?;
            if let Ok(rec) = postcard::from_bytes::<FileRec>(v.value()) { map.insert(k.value().to_string(), Arc::new(rec)); }
        }
        *self.files.write() = map;
        Ok(())
    }

    pub fn root(&self) -> &Path { &self.root }
    pub fn file_count(&self) -> usize { self.files.read().len() }
    pub fn files(&self) -> Vec<(String, Arc<FileRec>)> { self.files.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect() }
    pub fn get(&self, rel: &str) -> Option<Arc<FileRec>> { self.files.read().get(rel).cloned() }

    pub fn rel(&self, p: &Path) -> String {
        p.strip_prefix(&self.root).unwrap_or(p).to_string_lossy().replace('\\', "/")
    }

    fn lang_enabled(&self, l: Language) -> bool { self.languages.iter().any(|s| s == l.name()) }

    fn walker(&self) -> ignore::WalkBuilder {
        let mut wb = ignore::WalkBuilder::new(&self.root);
        wb.hidden(true).git_ignore(true).git_global(true).follow_links(false).max_depth(Some(40));
        let mut ob = ignore::overrides::OverrideBuilder::new(&self.root);
        for e in &self.exclude { let _ = ob.add(&format!("!{e}")); }
        if let Ok(o) = ob.build() { wb.overrides(o); }
        wb
    }

    /// Incremental scan: parse files whose mtime/size changed; remove vanished files.
    pub fn scan(&self) -> Result<ScanStats> {
        let t0 = Instant::now();
        let mut stats = ScanStats::default();
        let mut seen: Vec<(String, PathBuf, u64, u64, Language)> = Vec::new();
        for e in self.walker().build().flatten() {
            if !e.file_type().map(|t| t.is_file()).unwrap_or(false) { continue; }
            let Some(lang) = Language::from_path(e.path()) else { continue };
            if !self.lang_enabled(lang) { continue; }
            let Ok(md) = e.metadata() else { continue };
            if md.len() > self.max_file_bytes { continue; }
            let mtime = md.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0);
            seen.push((self.rel(e.path()), e.into_path(), mtime, md.len(), lang));
        }
        stats.files_seen = seen.len();
        let current = self.files.read().clone();
        let todo: Vec<_> = seen.iter().filter(|(rel, _, mtime, size, _)| current.get(rel).map(|r| r.mtime_ns != *mtime || r.size != *size).unwrap_or(true)).cloned().collect();
        stats.unchanged = seen.len() - todo.len();

        let parsed: Vec<(String, FileRec)> = todo.par_iter().filter_map(|(rel, path, mtime, size, lang)| {
            let bytes = std::fs::read(path).ok()?;
            let hash = *blake3::hash(&bytes).as_bytes();
            if let Some(old) = current.get(rel) { if old.hash == hash { return Some((rel.clone(), FileRec { mtime_ns: *mtime, size: *size, ..(**old).clone() })); } }
            let p = parse::parse(*lang, &bytes);
            let mut refs: Vec<(SmolStr, u16)> = p.refs.into_iter().collect();
            refs.sort_by(|a, b| b.1.cmp(&a.1));
            refs.truncate(2000);
            Some((rel.clone(), FileRec { mtime_ns: *mtime, size: *size, hash, lang: *lang, symbols: p.symbols, refs }))
        }).collect();
        stats.parsed = parsed.len();

        let seen_set: std::collections::HashSet<&str> = seen.iter().map(|s| s.0.as_str()).collect();
        let removed: Vec<String> = current.keys().filter(|k| !seen_set.contains(k.as_str())).cloned().collect();
        stats.removed = removed.len();

        if !parsed.is_empty() || !removed.is_empty() {
            let w = self.db.begin_write()?;
            {
                let mut t = w.open_table(FILES)?;
                for k in &removed { t.remove(k.as_str())?; }
                for (k, rec) in &parsed { t.insert(k.as_str(), postcard::to_allocvec(rec)?.as_slice())?; }
            }
            w.commit()?;
            let mut map = self.files.write();
            for k in &removed { map.remove(k); }
            for (k, rec) in parsed { map.insert(k, Arc::new(rec)); }
        }
        stats.symbols = self.files.read().values().map(|r| r.symbols.len()).sum();
        stats.ms = t0.elapsed().as_millis();
        tracing::info!(?stats, "index scan");
        Ok(stats)
    }

    /// Re-index specific files (after edits). Cheap and synchronous.
    pub fn update_paths(&self, paths: &[PathBuf]) -> Result<()> {
        let mut changed = Vec::new();
        for p in paths {
            let Some(lang) = Language::from_path(p) else { continue };
            if !self.lang_enabled(lang) { continue; }
            let rel = self.rel(p);
            let Ok(bytes) = std::fs::read(p) else { continue };
            let md = std::fs::metadata(p)?;
            let mtime = md.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0);
            let parsed = parse::parse(lang, &bytes);
            let mut refs: Vec<(SmolStr, u16)> = parsed.refs.into_iter().collect();
            refs.sort_by(|a, b| b.1.cmp(&a.1));
            refs.truncate(2000);
            changed.push((rel, FileRec { mtime_ns: mtime, size: md.len(), hash: *blake3::hash(&bytes).as_bytes(), lang, symbols: parsed.symbols, refs }));
        }
        if changed.is_empty() { return Ok(()); }
        let w = self.db.begin_write()?;
        { let mut t = w.open_table(FILES)?; for (k, rec) in &changed { t.insert(k.as_str(), postcard::to_allocvec(rec)?.as_slice())?; } }
        w.commit()?;
        let mut map = self.files.write();
        for (k, rec) in changed { map.insert(k, Arc::new(rec)); }
        Ok(())
    }

    pub fn outline(&self, p: &Path) -> Result<Vec<Symbol>> {
        let rel = self.rel(p);
        if let Some(r) = self.get(&rel) { return Ok(r.symbols.clone()); }
        let lang = Language::from_path(p).context("unsupported file type")?;
        let bytes = std::fs::read(p)?;
        Ok(parse::parse(lang, &bytes).symbols)
    }

    /// Fuzzy symbol search across the index.
    pub fn search(&self, query: &str, kind: Option<SymKind>, limit: usize) -> Vec<SymbolHit> {
        use nucleo_matcher::{pattern::{CaseMatching, Normalization, Pattern}, Config, Matcher, Utf32Str};
        let mut matcher = Matcher::new(Config::DEFAULT);
        let pat = Pattern::parse(query, CaseMatching::Smart, Normalization::Smart);
        let files = self.files.read();
        let mut hits: Vec<SymbolHit> = Vec::new();
        let mut buf = Vec::new();
        for (path, rec) in files.iter() {
            for s in &rec.symbols {
                if let Some(k) = kind { if s.kind != k { continue; } }
                let hay = Utf32Str::new(&s.name, &mut buf);
                if let Some(score) = pat.score(hay, &mut matcher) {
                    let exact_bonus = if s.name.eq_ignore_ascii_case(query) { 1000 } else { 0 };
                    hits.push(SymbolHit { score: score + exact_bonus + if s.is_public { 5 } else { 0 }, path: path.clone(), symbol: s.clone() });
                }
            }
        }
        hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
        hits.truncate(limit);
        hits
    }

    /// Languages by file count (for the system prompt).
    pub fn language_summary(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for r in self.files.read().values() { if r.lang.is_code() { *counts.entry(r.lang.name()).or_default() += 1; } }
        let mut v: Vec<(String, usize)> = counts.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

#[derive(Debug, Clone)]
pub struct SymbolHit { pub score: u32, pub path: String, pub symbol: Symbol }

pub fn format_outline(rel: &str, symbols: &[Symbol]) -> String {
    let mut s = format!("{rel}:\n");
    if symbols.is_empty() { s.push_str("  (no symbols)\n"); return s; }
    for sym in symbols {
        let depth = { let mut d = 0; let mut p = sym.parent; while let Some(i) = p { d += 1; p = symbols.get(i as usize).and_then(|x| x.parent); if d > 8 { break; } } d };
        let label = sym.kind.label();
        let sig_has_kw = sym.signature.split(|c: char| !c.is_alphanumeric() && c != '_').any(|w| w == label || (label == "fn" && matches!(w, "def" | "function" | "func")) || (label == "mod" && matches!(w, "namespace" | "module")));
        if sig_has_kw { s.push_str(&format!("  {:>5} {}{}\n", sym.line_start, "  ".repeat(depth), sym.signature)); }
        else { s.push_str(&format!("  {:>5} {}{} {}\n", sym.line_start, "  ".repeat(depth), label, sym.signature)); }
    }
    s
}
