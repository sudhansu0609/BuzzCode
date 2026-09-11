//! rmcp-backed MCP client + Tool adapter.

use anyhow::{Context, Result};
use cb_config::McpServer as McpServerConfig;
use cb_tool_api::*;
use rmcp::model::{CallToolRequestParams, ContentBlock};
use rmcp::service::RunningService;
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

pub struct McpClient {
    pub name: String,
    cfg: McpServerConfig,
    service: parking_lot::Mutex<Option<Arc<RunningService<RoleClient, ()>>>>,
}

impl McpClient {
    pub async fn connect(cfg: &McpServerConfig) -> Result<Arc<Self>> {
        let me = Arc::new(Self { name: cfg.name.clone(), cfg: cfg.clone(), service: parking_lot::Mutex::new(None) });
        me.ensure().await?;
        Ok(me)
    }

    async fn spawn(cfg: &McpServerConfig) -> Result<RunningService<RoleClient, ()>> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args);
        for (k, v) in &cfg.env { cmd.env(k, v); }
        #[cfg(windows)]
        { cmd.creation_flags(0x0800_0000); }
        let transport = TokioChildProcess::new(cmd).with_context(|| format!("spawning MCP server `{}` ({})", cfg.name, cfg.command))?;
        let service = tokio::time::timeout(Duration::from_secs(cfg.timeout_s.max(5)), ().serve(transport)).await
            .map_err(|_| anyhow::anyhow!("MCP server `{}` did not finish initialize within {}s", cfg.name, cfg.timeout_s))?
            .with_context(|| format!("MCP initialize failed for `{}`", cfg.name))?;
        Ok(service)
    }

    async fn ensure(&self) -> Result<Arc<RunningService<RoleClient, ()>>> {
        if let Some(s) = self.service.lock().clone() { return Ok(s); }
        let s = Arc::new(Self::spawn(&self.cfg).await?);
        *self.service.lock() = Some(s.clone());
        Ok(s)
    }

    fn drop_service(&self) { *self.service.lock() = None; }

    pub async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>> {
        let s = self.ensure().await?;
        Ok(s.list_all_tools().await.context("tools/list")?)
    }

    pub async fn call(&self, name: &str, args: Value, timeout: Duration) -> Result<rmcp::model::CallToolResult> {
        let s = self.ensure().await?;
        let mut params = CallToolRequestParams::new(name.to_string());
        if let Value::Object(m) = args { params = params.with_arguments(m); }
        match tokio::time::timeout(timeout, s.call_tool(params)).await {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => {
                // Transport errors usually mean the server died: reconnect next time.
                if matches!(e, rmcp::ServiceError::TransportSend(_) | rmcp::ServiceError::TransportClosed) { self.drop_service(); }
                Err(anyhow::anyhow!("MCP call {}/{name}: {e}", self.name))
            }
            Err(_) => Err(anyhow::anyhow!("MCP call {}/{name} timed out after {}s", self.name, timeout.as_secs())),
        }
    }
}

/// A remote MCP tool exposed as a local tool named `mcp__<server>__<tool>`.
pub struct McpTool {
    client: Arc<McpClient>,
    remote_name: String,
    spec: ToolSpec,
    timeout: Duration,
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    async fn call(&self, args: Value, _cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let r = self.client.call(&self.remote_name, args, self.timeout).await.map_err(|e| ToolError::Failed(format!("{e:#}")))?;
        let mut text = String::new();
        for c in &r.content {
            match c {
                ContentBlock::Text(t) => { text.push_str(&t.text); text.push('\n'); }
                ContentBlock::Image(_) => text.push_str("[image omitted]\n"),
                ContentBlock::Audio(_) => text.push_str("[audio omitted]\n"),
                ContentBlock::Resource(res) => text.push_str(&format!("[resource {}]\n", serde_json::to_string(&res.resource).unwrap_or_default().chars().take(2000).collect::<String>())),
                ContentBlock::ResourceLink(l) => text.push_str(&format!("[resource link {}]\n", l.uri)),
                _ => text.push_str("[unsupported content omitted]\n"),
            }
        }
        if let Some(sc) = &r.structured_content { if text.trim().is_empty() { text = serde_json::to_string_pretty(sc).unwrap_or_default(); } }
        let mut out = ToolOutput::text(text.trim_end().to_string());
        out.is_error = r.is_error.unwrap_or(false);
        Ok(out)
    }
}

fn class_for(trust: &str) -> PermissionClass {
    match trust { "readonly" => PermissionClass::ReadOnly, "yolo" => PermissionClass::ReadOnly, _ => PermissionClass::Network }
}

/// Connect every enabled server and return tools ready for registration. Failures are logged, not fatal.
pub async fn connect_all(servers: &[McpServerConfig]) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for cfg in servers.iter().filter(|s| s.enabled && !s.command.is_empty()) {
        match McpClient::connect(cfg).await {
            Ok(client) => match client.list_tools().await {
                Ok(list) => {
                    for t in list {
                        let name = format!("mcp__{}__{}", cfg.name, t.name);
                        let mut schema = Value::Object((*t.input_schema).clone());
                        if let Value::Object(m) = &mut schema { m.remove("$schema"); }
                        let spec = ToolSpec { name: name.clone(), description: format!("[MCP {}] {}", cfg.name, t.description.as_deref().unwrap_or("")), schema, class: class_for(&cfg.trust) };
                        tools.push(Arc::new(McpTool { client: client.clone(), remote_name: t.name.to_string(), spec, timeout: Duration::from_secs(cfg.timeout_s.max(5)) }));
                    }
                    tracing::info!(server = %cfg.name, n = tools.len(), "MCP tools registered");
                }
                Err(e) => tracing::warn!(server = %cfg.name, "MCP tools/list failed: {e:#}"),
            },
            Err(e) => tracing::warn!(server = %cfg.name, "MCP connect failed: {e:#}"),
        }
    }
    tools
}
