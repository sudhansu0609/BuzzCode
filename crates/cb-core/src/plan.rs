//! Plan mode: the `write_plan` tool + persisted plan documents.

use cb_tool_api::*;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PlanStep {
    /// Step number, starting at 1.
    pub n: u32,
    /// What to do, concretely.
    pub description: String,
    /// Files to create or modify.
    #[serde(default)]
    pub files: Vec<String>,
    /// How to verify this step (command or check).
    #[serde(default)]
    pub verification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Plan {
    /// Short title.
    pub title: String,
    /// 2-5 sentence summary of the approach and why.
    pub summary: String,
    /// Ordered steps.
    pub steps: Vec<PlanStep>,
    /// Risks or things that might go wrong.
    #[serde(default)]
    pub risks: Vec<String>,
    /// Questions the user should answer before implementation.
    #[serde(default)]
    pub questions: Vec<String>,
}

impl Plan {
    pub fn slug(&self) -> String {
        let s: String = self.title.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
        let s = s.trim_matches('-').split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
        if s.is_empty() { "plan".into() } else { s.chars().take(48).collect() }
    }

    pub fn to_markdown(&self) -> String {
        let mut m = format!("# {}\n\n{}\n\n## Steps\n", self.title, self.summary);
        for s in &self.steps {
            m.push_str(&format!("\n{}. {}\n", s.n, s.description));
            if !s.files.is_empty() { m.push_str(&format!("   - files: {}\n", s.files.join(", "))); }
            if !s.verification.is_empty() { m.push_str(&format!("   - verify: {}\n", s.verification)); }
        }
        if !self.risks.is_empty() { m.push_str("\n## Risks\n"); for r in &self.risks { m.push_str(&format!("- {r}\n")); } }
        if !self.questions.is_empty() { m.push_str("\n## Open questions\n"); for q in &self.questions { m.push_str(&format!("- {q}\n")); } }
        m
    }
}

/// Session-level holder of the latest plan (read by the TUI for `/act`).
#[derive(Default)]
pub struct PlanStore {
    pub latest: Mutex<Option<(Plan, PathBuf)>>,
}

pub struct WritePlan { pub store: Arc<PlanStore>, pub plans_dir: PathBuf }

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Plan>(
    "write_plan",
    "Submit the final implementation plan (call exactly once, after investigating). Steps must name concrete files and a verification for each.",
    PermissionClass::ReadOnly,
));

#[async_trait::async_trait]
impl Tool for WritePlan {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn call(&self, args: Value, _cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let plan: Plan = parse_args(&args)?;
        if plan.steps.is_empty() { return Ok(ToolOutput::error("plan has no steps; add at least one concrete step")); }
        std::fs::create_dir_all(&self.plans_dir)?;
        let base = self.plans_dir.join(plan.slug());
        let md = base.with_extension("md");
        let json = base.with_extension("json");
        std::fs::write(&md, plan.to_markdown())?;
        std::fs::write(&json, serde_json::to_string_pretty(&plan).unwrap_or_default())?;
        *self.store.latest.lock() = Some((plan.clone(), md.clone()));
        Ok(ToolOutput::text(format!("Plan saved to {} ({} steps). Tell the user the plan is ready for review; do not start implementing.", md.display(), plan.steps.len())))
    }
}
