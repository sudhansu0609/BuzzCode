//! Builtin tools.

pub mod edit_file;
pub mod fsutil;
pub mod git;
pub mod glob;
pub mod grep;
pub mod list_dir;
pub mod read_file;
pub mod shell;
pub mod write_file;

use cb_tool_api::Tool;
use std::sync::Arc;

/// All builtin tools in registration order.
pub fn builtin_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(read_file::ReadFile),
        Arc::new(write_file::WriteFile),
        Arc::new(edit_file::EditFile),
        Arc::new(glob::Glob),
        Arc::new(grep::Grep),
        Arc::new(list_dir::ListDir),
        Arc::new(shell::Shell),
        Arc::new(git::GitStatus),
        Arc::new(git::GitDiff),
        Arc::new(git::GitLog),
        Arc::new(git::GitCommit),
    ]
}

/// Names of tools that never modify anything.
pub const READ_ONLY_TOOLS: &[&str] = &["read_file", "glob", "grep", "list_dir", "git_status", "git_diff", "git_log", "outline", "symbol_search", "repo_map"];
