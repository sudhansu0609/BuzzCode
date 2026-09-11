//! OpenAI-compatible client for llama-server: streaming chat, tokenize, health, slots, metrics.

use anyhow::{bail, Context, Result};
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::pin::Pin;
use std::time::Duration;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Timings {
    pub prompt_n: u32,
    pub cache_n: u32,
    pub prompt_ms: f64,
    pub predicted_n: u32,
    pub predicted_ms: f64,
    pub draft_n: u32,
    pub draft_n_accepted: u32,
}

impl Timings {
    pub fn prompt_tps(&self) -> f64 { if self.prompt_ms > 0.0 { self.prompt_n as f64 * 1000.0 / self.prompt_ms } else { 0.0 } }
    pub fn decode_tps(&self) -> f64 { if self.predicted_ms > 0.0 { self.predicted_n as f64 * 1000.0 / self.predicted_ms } else { 0.0 } }
    /// Fraction of the prompt served from cache. 1.0 = perfect prefix reuse.
    pub fn cache_ratio(&self) -> f64 {
        let total = self.prompt_n + self.cache_n;
        if total == 0 { 0.0 } else { self.cache_n as f64 / total as f64 }
    }
    pub fn draft_acceptance(&self) -> Option<f64> {
        if self.draft_n > 0 { Some(self.draft_n_accepted as f64 / self.draft_n as f64) } else { None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason { Stop, Length, ToolCalls, Other(String) }

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Reasoning(String),
    Content(String),
    ToolCallDelta { index: u32, id: Option<String>, name: Option<String>, args_fragment: String },
    Usage { prompt_tokens: u32, completion_tokens: u32 },
    Timings(Timings),
    Done(FinishReason),
    Error(String),
}

/// Request body for `/v1/chat/completions`. Built by the agent; everything is explicit so the
/// serialized bytes are deterministic (prefix stability).
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub timings_per_token: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_probs: Option<u32>,
}

impl ChatRequest {
    pub fn new(model: &str, messages: Vec<Value>) -> Self {
        Self {
            model: model.into(), messages, tools: None, tool_choice: None, response_format: None,
            stream: true, cache_prompt: Some(true), id_slot: Some(0), max_tokens: None,
            temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 0.0, repeat_penalty: 1.0,
            chat_template_kwargs: None, stop: None, timings_per_token: false, n_probs: None,
        }
    }
    /// Qwen3.8 template facts (verified via /apply-template):
    /// * valid `reasoning_effort`: `xhigh` (default), `medium`, `low`; anything else raises.
    /// * `xhigh` and `low` inject a sentence at the START of the system message (prefix break);
    ///   `medium` injects nothing.
    /// * `enable_thinking:false` only changes the tail (`<think>\n\n</think>`), so
    ///   "none" = medium + enable_thinking:false is cache-safe relative to a medium session.
    pub fn reasoning_effort(mut self, effort: &str) -> Self {
        let mut kw = self.chat_template_kwargs.take().unwrap_or_else(|| json!({}));
        if let Value::Object(m) = &mut kw {
            let (eff, think) = match effort {
                "none" | "off" => ("medium", false),
                "xhigh" | "max" | "high" => ("xhigh", true),
                "low" => ("low", true),
                _ => ("medium", true),
            };
            m.insert("reasoning_effort".into(), Value::String(eff.into()));
            if !think { m.insert("enable_thinking".into(), Value::Bool(false)); }
        }
        self.chat_template_kwargs = Some(kw);
        self
    }
}

pub type ChatStream = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerProps {
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub total_slots: u32,
    #[serde(default)]
    pub chat_template: String,
    #[serde(default)]
    pub default_generation_settings: Value,
    #[serde(default)]
    pub build_info: String,
}

impl ServerProps {
    pub fn n_ctx(&self) -> u32 {
        self.default_generation_settings.get("n_ctx").and_then(Value::as_u64).unwrap_or(0) as u32
    }
}

#[derive(Debug, Clone)]
pub struct LlamaClient {
    http: reqwest::Client,
    base: String,
    api_key: Option<String>,
}

impl LlamaClient {
    pub fn new(base_url: &str) -> Self {
        Self::with_api_key(base_url, None)
    }

    pub fn with_api_key(base_url: &str, api_key: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .tcp_nodelay(true)
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("reqwest client");
        Self { http, base: base_url.trim_end_matches('/').to_string(), api_key }
    }

    pub fn base_url(&self) -> &str { &self.base }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(key) = &self.api_key {
            req.header("Authorization", format!("Bearer {key}"))
        } else {
            req
        }
    }

    pub async fn health(&self) -> Result<bool> {
        // 1. Try standard /health endpoint (native to llama-server)
        let r = self.apply_auth(self.http.get(format!("{}/health", self.base))).timeout(Duration::from_secs(3)).send().await;
        if let Ok(resp) = r {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    if !text.contains("Unexpected endpoint") && !text.contains("\"error\"") {
                        return Ok(true);
                    }
                } else {
                    return Ok(true);
                }
            }
        }
        // 2. Fallback to /v1/models or /models (LM Studio, Ollama, OpenAI-compatible APIs)
        let models_url = if self.base.ends_with("/v1") {
            format!("{}/models", self.base)
        } else {
            format!("{}/v1/models", self.base)
        };
        let r2 = self.apply_auth(self.http.get(&models_url)).timeout(Duration::from_secs(3)).send().await;
        if let Ok(resp) = r2 {
            if resp.status().is_success() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub async fn props(&self) -> Result<ServerProps> {
        let resp = self.apply_auth(self.http.get(format!("{}/props", self.base)))
            .timeout(Duration::from_secs(5))
            .send().await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let text = r.text().await.unwrap_or_default();
                if let Ok(p) = serde_json::from_str::<ServerProps>(&text) {
                    return Ok(p);
                }
            }
            _ => {}
        }
        Ok(ServerProps::default())
    }

    /// Exact token count via the server's tokenizer.
    pub async fn tokenize_count(&self, content: &str) -> Result<u32> {
        let v: Value = self.http.post(format!("{}/tokenize", self.base))
            .json(&json!({ "content": content, "add_special": false, "with_pieces": false }))
            .timeout(Duration::from_secs(60))
            .send().await?.error_for_status()?.json().await?;
        Ok(v.get("tokens").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0) as u32)
    }

    /// Render the chat template server-side (for prefix-stability checks in `doctor`).
    pub async fn apply_template(&self, messages: &[Value], tools: Option<&Value>, kwargs: Option<&Value>) -> Result<String> {
        let mut body = json!({ "messages": messages });
        if let Some(t) = tools { body["tools"] = t.clone(); }
        if let Some(k) = kwargs { body["chat_template_kwargs"] = k.clone(); }
        let v: Value = self.http.post(format!("{}/apply-template", self.base)).json(&body).timeout(Duration::from_secs(30))
            .send().await?.error_for_status()?.json().await?;
        Ok(v.get("prompt").and_then(Value::as_str).unwrap_or("").to_string())
    }

    pub async fn slot_action(&self, id: u32, action: &str, filename: Option<&str>) -> Result<Value> {
        let mut req = self.http.post(format!("{}/slots/{id}?action={action}", self.base)).timeout(Duration::from_secs(120));
        if let Some(f) = filename { req = req.json(&json!({ "filename": f })); } else { req = req.json(&json!({})); }
        let resp = req.send().await?;
        let status = resp.status();
        let v: Value = resp.json().await.unwrap_or(Value::Null);
        if !status.is_success() { bail!("slot {action} failed ({status}): {v}"); }
        Ok(v)
    }
    pub async fn slot_save(&self, id: u32, filename: &str) -> Result<Value> { self.slot_action(id, "save", Some(filename)).await }
    pub async fn slot_restore(&self, id: u32, filename: &str) -> Result<Value> { self.slot_action(id, "restore", Some(filename)).await }
    pub async fn slot_erase(&self, id: u32) -> Result<Value> { self.slot_action(id, "erase", None).await }

    pub async fn slots(&self) -> Result<Value> {
        Ok(self.http.get(format!("{}/slots", self.base)).timeout(Duration::from_secs(5)).send().await?.error_for_status()?.json().await?)
    }

    pub async fn metrics_raw(&self) -> Result<String> {
        Ok(self.http.get(format!("{}/metrics", self.base)).timeout(Duration::from_secs(5)).send().await?.error_for_status()?.text().await?)
    }

    /// Streaming chat completion. The returned stream yields parsed events and ends after `Done`/`Error`.
    pub async fn chat_stream(&self, req: &ChatRequest) -> Result<ChatStream> {
        let url = if self.base.ends_with("/v1") {
            format!("{}/chat/completions", self.base)
        } else {
            format!("{}/v1/chat/completions", self.base)
        };
        let rb = self.apply_auth(self.http.post(&url).json(req).timeout(Duration::from_secs(3600)));
        let resp = rb.send().await.context("POST /v1/chat/completions")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("chat completion failed ({status}): {}", body.chars().take(2000).collect::<String>());
        }
        let es = resp.bytes_stream().eventsource();
        let stream = es.flat_map(|item| {
            let events: Vec<StreamEvent> = match item {
                Ok(ev) => crate::sse::parse_chunk(&ev.data),
                Err(e) => vec![StreamEvent::Error(format!("stream error: {e}"))],
            };
            futures::stream::iter(events)
        });
        Ok(Box::pin(stream))
    }

    /// Non-streaming convenience used by doctor/bench: returns (content, reasoning, timings).
    pub async fn chat_once(&self, mut req: ChatRequest) -> Result<(String, String, Timings, Vec<crate::sse::ToolCallAcc>)> {
        req.stream = true;
        let mut s = self.chat_stream(&req).await?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut timings = Timings::default();
        let mut calls: Vec<crate::sse::ToolCallAcc> = Vec::new();
        while let Some(ev) = s.next().await {
            match ev {
                StreamEvent::Content(t) => content.push_str(&t),
                StreamEvent::Reasoning(t) => reasoning.push_str(&t),
                StreamEvent::Timings(t) => timings = t,
                StreamEvent::ToolCallDelta { index, id, name, args_fragment } => {
                    let acc = match calls.iter_mut().find(|c| c.index == index) {
                        Some(c) => c,
                        None => { calls.push(crate::sse::ToolCallAcc { index, ..Default::default() }); calls.last_mut().unwrap() }
                    };
                    if let Some(id) = id { acc.id = id; }
                    if let Some(n) = name { acc.name.push_str(&n); }
                    acc.arguments.push_str(&args_fragment);
                }
                StreamEvent::Error(e) => bail!("{e}"),
                StreamEvent::Done(_) | StreamEvent::Usage { .. } => {}
            }
        }
        Ok((content, reasoning, timings, calls))
    }
}
