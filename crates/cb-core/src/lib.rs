//! Agent core: messages, prefix-stable context, tool registry, parser, permissions, agent loop.

pub mod agent;
pub mod context;
pub mod events;
pub mod guard;
pub mod handle;
pub mod message;
pub mod parser;
pub mod permission;
pub mod plan;
pub mod prompt;
pub mod registry;
pub mod session;
pub mod subagent;
pub mod task;
pub mod tokens;
pub mod truncate;

pub use agent::{Agent, AgentOutcome};
pub use context::{ContextStore, FrozenPrefix};
pub use events::AgentEvent;
pub use handle::{AgentCmd, AgentFactory, AgentHandle, AgentMode, AgentStats};
pub use message::{Message, Role, ToolCall};
pub use permission::{PermissionBroker, PermissionDecision, PermissionRequest};
pub use registry::ToolRegistry;
pub use session::Session;
pub use task::TaskClass;
pub use tokens::TokenCounter;

pub use cb_tool_api::{PermissionClass, PermissionMode, Tool, ToolCtx, ToolOutput, ToolSpec};
