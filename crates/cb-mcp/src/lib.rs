//! MCP client (stdio transport) exposing remote tools as `cb_tool_api::Tool`s.
pub mod client;
pub use client::{connect_all, McpClient, McpTool};
