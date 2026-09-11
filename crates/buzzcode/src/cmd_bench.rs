//! `buzzcode bench engine|cache|eval` and `buzzcode engine tune`.

use anyhow::{Context, Result};
use cb_config::Config;
use cb_engine::client::ChatRequest;
use cb_engine::{nvidia, EngineManager, Timings};
use clap::Subcommand;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Subcommand, Debug)]
pub enum BenchCmd {
    /// Prompt/decode throughput, TTFT, MTP acceptance, VRAM and RSS for the active profile.
    Engine {
        /// Prompt sizes (tokens) to measure prompt-processing at.
        #[arg(long, default_values_t = [2048u32, 8192, 16384])]
        prompt_sizes: Vec<u32>,
        #[arg(long, default_value_t = 256)]
        gen_tokens: u32,
    },
    /// Multi-turn prefix-cache regression: cache_n/prompt_n per turn must stay high.
    Cache { #[arg(long, default_value_t = 8)] turns: u32 },
    /// Run the local eval tasks (eval/tasks/<name>/{repo,task.md,check.ps1}).
    Eval {
        /// Only run tasks whose name contains this.
        #[arg(long)] filter: Option<String>,
    },
}

fn synthetic_code(tokens_approx: u32) -> String {
    // ~3.3 chars/token for code; deterministic content so the bench is reproducible.
    let mut s = String::new();
    let mut i = 0;
    while (s.len() as f32 / 3.3) < tokens_approx as f32 {
        s.push_str(&format!("pub fn compute_{i}(input: &[u32], scale: u32) -> Result<Vec<u32>, String> {{\n    if input.is_empty() {{ return Err(\"empty input {i}\".to_string()); }}\n    let mut out = Vec::with_capacity(input.len());\n    for (idx, v) in input.iter().enumerate() {{\n        out.push(v.wrapping_mul(scale).wrapping_add(idx as u32 + {i}));\n    }}\n    Ok(out)\n}}\n\n"));
        i += 1;
    }
    s
}

fn rss_mib() -> u64 {
    #[cfg(windows)]
    {
        let pid = std::process::id();
        let out = std::process::Command::new("powershell").args(["-NoProfile", "-NonInteractive", "-Command", &format!("(Get-Process -Id {pid}).WorkingSet64")]).output();
        if let Ok(o) = out { if let Ok(v) = String::from_utf8_lossy(&o.stdout).trim().parse::<u64>() { return v >> 20; } }
    }
    0
}

async fn start(cfg: &Arc<Config>) -> Result<Arc<EngineManager>> {
    let mgr = EngineManager::new(cfg.clone(), cfg.active_profile().clone());
    mgr.start().await?;
    Ok(mgr)
}

pub async fn run(cfg: Config, cmd: BenchCmd) -> Result<()> {
    let cfg = Arc::new(cfg);
    std::fs::create_dir_all(cfg.paths.bench_dir())?;
    match cmd {
        BenchCmd::Engine { prompt_sizes, gen_tokens } => bench_engine(cfg, prompt_sizes, gen_tokens).await,
        BenchCmd::Cache { turns } => bench_cache(cfg, turns).await,
        BenchCmd::Eval { filter } => bench_eval(cfg, filter).await,
    }
}

async fn bench_engine(cfg: Arc<Config>, prompt_sizes: Vec<u32>, gen_tokens: u32) -> Result<()> {
    let gpu0 = nvidia::query().ok();
    let mgr = start(&cfg).await?;
    let alias = mgr.alias().to_string();
    let client = mgr.client().clone();
    let mut rows: Vec<Value> = Vec::new();
    println!("profile {} · model {} · ctx {}", cfg.active_profile().name, mgr.model_path().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).unwrap_or_default(), mgr.n_ctx());
    println!("{:>8} {:>10} {:>10} {:>9} {:>9} {:>8}", "prompt", "pp tok/s", "tg tok/s", "ttft ms", "mtp %", "cached");
    for &n in &prompt_sizes {
        if n + gen_tokens + 64 > mgr.n_ctx() { println!("{n:>8}  (skipped: exceeds ctx)"); continue; }
        let code = synthetic_code(n);
        let msgs = vec![json!({"role":"system","content":"You are a code review assistant."}), json!({"role":"user","content": format!("Review this code and list the three most important issues:\n```rust\n{code}\n```")})];
        // Cold: erase slot so the prompt is fully processed.
        let _ = client.slot_erase(0).await;
        let mut req = ChatRequest::new(&alias, msgs.clone());
        req.max_tokens = Some(gen_tokens);
        req = req.reasoning_effort("none");
        let t0 = Instant::now();
        let mut stream = client.chat_stream(&req).await?;
        let mut first: Option<f64> = None;
        let mut timings = Timings::default();
        use futures::StreamExt;
        while let Some(ev) = stream.next().await {
            match ev {
                cb_engine::StreamEvent::Content(_) | cb_engine::StreamEvent::Reasoning(_) => { if first.is_none() { first = Some(t0.elapsed().as_secs_f64() * 1000.0); } }
                cb_engine::StreamEvent::Timings(t) => timings = t,
                cb_engine::StreamEvent::Error(e) => anyhow::bail!("{e}"),
                _ => {}
            }
        }
        let ttft = first.unwrap_or(0.0);
        println!("{:>8} {:>10.0} {:>10.1} {:>9.0} {:>9} {:>8}", timings.prompt_n + timings.cache_n, timings.prompt_tps(), timings.decode_tps(), ttft, timings.draft_acceptance().map(|a| format!("{:.0}", a * 100.0)).unwrap_or("-".into()), timings.cache_n);
        rows.push(json!({"prompt_tokens": timings.prompt_n + timings.cache_n, "prompt_tps": timings.prompt_tps(), "decode_tps": timings.decode_tps(), "ttft_ms": ttft, "mtp_accept": timings.draft_acceptance(), "predicted_n": timings.predicted_n}));
        // Warm repeat (same prompt → cached prefix): TTFT should collapse.
        let t1 = Instant::now();
        let mut req2 = ChatRequest::new(&alias, msgs);
        req2.max_tokens = Some(16); req2 = req2.reasoning_effort("none");
        let (_, _, t2, _) = client.chat_once(req2).await?;
        println!("{:>8} {:>10} {:>10} {:>9.0} {:>9} {:>8}  (warm repeat)", "", "", "", t1.elapsed().as_secs_f64() * 1000.0, "", t2.cache_n);
    }
    let gpu1 = nvidia::query().ok();
    let vram = match (gpu0, gpu1) { (Some(a), Some(b)) => b.mem.used_mib.saturating_sub(a.mem.used_mib), _ => 0 };
    let rss = rss_mib();
    println!("server VRAM {vram} MiB · harness RSS {rss} MiB · slot switches {}", mgr.slot().switch_count());
    let report = json!({"profile": cfg.active_profile().name, "model": mgr.model_path(), "n_ctx": mgr.n_ctx(), "plan": mgr.plan(), "rows": rows, "vram_mib": vram, "rss_mib": rss, "build": client.props().await.map(|p| p.build_info).unwrap_or_default()});
    let out = cfg.paths.bench_dir().join(format!("engine-{}.json", cfg.active_profile().name));
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("saved {}", out.display());
    mgr.stop().await;
    Ok(())
}

async fn bench_cache(cfg: Arc<Config>, turns: u32) -> Result<()> {
    let mgr = start(&cfg).await?;
    let alias = mgr.alias().to_string();
    let client = mgr.client().clone();
    let system = json!({"role":"system","content": format!("You are buzzcode. Rules:\n{}", (1..=40).map(|i| format!("{i}. Read before editing; minimal exact edits; verify with tests.")).collect::<Vec<_>>().join("\n"))});
    let mut messages = vec![system];
    let mut prefix_hash: Option<String> = None;
    let mut min_ratio = 1.0f64;
    println!("{:>4} {:>8} {:>8} {:>8} {:>9}  prefix", "turn", "prompt_n", "cache_n", "cached%", "prompt ms");
    for t in 1..=turns {
        messages.push(json!({"role":"user","content": format!("Turn {t}: give me a one-line tip about Rust ownership, numbered {t}.")}));
        // Prefix stability check: the serialized messages minus the last one must start with the previous serialization.
        let ser = serde_json::to_string(&messages[..messages.len() - 1])?;
        let h = blake3_hex(&ser);
        let stable = prefix_hash.as_ref().map(|p| ser.starts_with(&serde_json::to_string(&messages[..messages.len() - 3.min(messages.len() - 1)]).unwrap_or_default().trim_end_matches(']').to_string()) || p == &h).unwrap_or(true);
        let mut req = ChatRequest::new(&alias, messages.clone());
        req.max_tokens = Some(48); req = req.reasoning_effort("none");
        let (content, _, ti, _) = client.chat_once(req).await?;
        let ratio = if t == 1 { 1.0 } else { ti.cache_ratio() };
        if t > 1 { min_ratio = min_ratio.min(ratio); }
        println!("{t:>4} {:>8} {:>8} {:>7.0}% {:>9.0}  {}", ti.prompt_n, ti.cache_n, ti.cache_ratio() * 100.0, ti.prompt_ms, if stable { "stable" } else { "CHANGED" });
        messages.push(json!({"role":"assistant","content": content}));
        prefix_hash = Some(h);
    }
    println!("min cached fraction after turn 1: {:.0}% → {}", min_ratio * 100.0, if min_ratio >= 0.85 { "PASS" } else { "FAIL (expected ≥ 85%)" });
    mgr.stop().await;
    if min_ratio < 0.85 { std::process::exit(1); }
    Ok(())
}

fn blake3_hex(s: &str) -> String { blake3::hash(s.as_bytes()).to_hex().to_string() }

async fn bench_eval(cfg: Arc<Config>, filter: Option<String>) -> Result<()> {
    let eval_dir = cfg.paths.project_dir.join(&cfg.bench.eval_dir);
    let mut tasks: Vec<PathBuf> = std::fs::read_dir(&eval_dir).with_context(|| format!("reading {}", eval_dir.display()))?
        .flatten().map(|e| e.path()).filter(|p| p.is_dir() && p.join("task.md").is_file()).collect();
    tasks.sort();
    if let Some(f) = &filter { tasks.retain(|p| p.file_name().unwrap().to_string_lossy().contains(f.as_str())); }
    let exe = std::env::current_exe()?;
    let mut results: Vec<Value> = Vec::new();
    println!("{:<28} {:>6} {:>6} {:>8}", "task", "pass", "turns", "wall s");
    for t in &tasks {
        let name = t.file_name().unwrap().to_string_lossy().to_string();
        let repo = t.join("repo");
        let tmp = t.join("repo-tmp");
        if tmp.exists() { std::fs::remove_dir_all(&tmp)?; }
        copy_dir(&repo, &tmp)?;
        let task = std::fs::read_to_string(t.join("task.md"))?;
        let t0 = Instant::now();
        let out = tokio::process::Command::new(&exe).args(["-C", &tmp.to_string_lossy(), "--mode", "yolo", "--json", "-p", task.trim()]).output().await?;
        let wall = t0.elapsed().as_secs_f64();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut turns = 0u32; let mut reason = String::from("?");
        for line in stdout.lines() {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if v["type"] == "finished" { turns = v["turns"].as_u64().unwrap_or(0) as u32; reason = v["reason"].as_str().unwrap_or("?").into(); }
            }
        }
        std::fs::write(t.join("last-run.jsonl"), stdout.as_bytes())?;
        let check = t.join("check.ps1");
        let pass = if check.is_file() {
            let st = tokio::process::Command::new("powershell").args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]).arg(&check).arg("-Dir").arg(&tmp).status().await?;
            st.success()
        } else { reason == "end_turn" };
        println!("{name:<28} {:>6} {turns:>6} {wall:>8.1}   ({reason})", if pass { "PASS" } else { "FAIL" });
        results.push(json!({"task": name, "pass": pass, "turns": turns, "wall_s": wall, "reason": reason}));
    }
    let passed = results.iter().filter(|r| r["pass"] == true).count();
    println!("{passed}/{} passed", results.len());
    let out = cfg.paths.bench_dir().join(format!("eval-{}.json", cfg.active_profile().name));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"profile": cfg.active_profile().name, "results": results}))?)?;
    println!("saved {}", out.display());
    Ok(())
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let p = e.path();
        let name = e.file_name();
        if name == "target" || name == "node_modules" || name == ".buzzcode" { continue; }
        let d = dst.join(&name);
        if p.is_dir() { copy_dir(&p, &d)?; } else { std::fs::copy(&p, &d)?; }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// engine tune
// ---------------------------------------------------------------------------------------------

/// Sweep speculative modes (and thread counts when CPU layers exist) and persist the best.
pub async fn tune(cfg: Config) -> Result<()> {
    let base = Arc::new(cfg);
    let profile = base.active_profile().clone();
    let specs = ["draft-mtp", "ngram-simple", "none"];
    let code_prompt = format!("Refactor this code for readability, output the full rewritten code:\n```rust\n{}\n```", synthetic_code(700));
    let mut best: Option<(String, f64)> = None;
    println!("{:<14} {:>10} {:>10} {:>8}", "spec", "tg tok/s", "pp tok/s", "mtp %");
    for spec in specs {
        let mut cfg = (*base).clone();
        let p = cfg.profiles.iter_mut().find(|p| p.name == profile.name).unwrap();
        p.spec = spec.into();
        if spec == "ngram-simple" { p.spec_draft_n_max = 64; }
        let cfg = Arc::new(cfg);
        let mgr = match start(&cfg).await { Ok(m) => m, Err(e) => { println!("{spec:<14} failed to start: {e:#}"); continue; } };
        let client = mgr.client().clone();
        let mut req = ChatRequest::new(&mgr.alias(), vec![json!({"role":"user","content": code_prompt.clone()})]);
        req.max_tokens = Some(400); req = req.reasoning_effort("none");
        let res = client.chat_once(req).await;
        mgr.stop().await;
        match res {
            Ok((_, _, t, _)) => {
                println!("{spec:<14} {:>10.1} {:>10.0} {:>8}", t.decode_tps(), t.prompt_tps(), t.draft_acceptance().map(|a| format!("{:.0}", a * 100.0)).unwrap_or("-".into()));
                if best.as_ref().map(|b| t.decode_tps() > b.1).unwrap_or(true) { best = Some((spec.into(), t.decode_tps())); }
            }
            Err(e) => println!("{spec:<14} error: {e:#}"),
        }
    }
    if let Some((spec, tps)) = best {
        println!("best: {spec} ({tps:.1} tok/s on a code-rewrite prompt)");
        // Persist into the tuned profile record.
        if let Some(model) = cb_engine::discover::find_model(&base, &profile.model) {
            let hash = cb_engine::tune::model_hash(&model)?;
            if let Some(mut t) = cb_engine::tune::load(&base, &hash, &profile.name) {
                t.spec = Some(spec.clone()); t.measured_decode_tps = Some(tps);
                cb_engine::tune::save(&base, &t)?;
                println!("recorded in {}", base.paths.tune_dir().display());
            }
        }
        if spec != profile.spec { println!("→ set `spec = \"{spec}\"` in your [[profile]] to use it by default"); }
    }
    Ok(())
}
