//! Task classes → sampling + reasoning effort.

use cb_config::{Profile, Sampling};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass { Chat, Plan, Explore, Edit, Summarize, CommitMsg }

impl TaskClass {
    pub fn sampling(self, p: &Profile) -> Sampling {
        match self {
            TaskClass::Plan | TaskClass::Explore => p.sampling.thinking,
            TaskClass::Chat | TaskClass::Edit => p.sampling.coding,
            TaskClass::Summarize | TaskClass::CommitMsg => p.sampling.instruct,
        }
    }

    pub fn effort<'a>(self, p: &'a Profile) -> &'a str {
        match self {
            TaskClass::Plan => &p.effort.plan,
            TaskClass::Explore => &p.effort.explore,
            TaskClass::Edit => &p.effort.edit,
            TaskClass::Chat => &p.effort.chat,
            TaskClass::Summarize => &p.effort.summarize,
            TaskClass::CommitMsg => &p.effort.commit,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self { TaskClass::Chat => "chat", TaskClass::Plan => "plan", TaskClass::Explore => "explore", TaskClass::Edit => "edit", TaskClass::Summarize => "summarize", TaskClass::CommitMsg => "commit" }
    }
}

/// Normalize user-facing effort strings.
pub fn normalize_effort(s: &str) -> Option<&'static str> {
    match s.trim().to_ascii_lowercase().as_str() {
        "xhigh" | "max" | "high" => Some("xhigh"), "medium" | "med" => Some("medium"),
        "low" => Some("low"), "none" | "off" | "no" => Some("none"), _ => None,
    }
}
