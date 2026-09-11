//! Engine management: llama.cpp discovery/build, GGUF inspection, VRAM planning,
//! llama-server process supervision and the OpenAI-compatible streaming client.

pub mod build;
pub mod catalog;
pub mod client;
pub mod discover;
pub mod download;
pub mod gguf;
pub mod manager;
pub mod nvidia;
pub mod pidfile;
pub mod ports;
pub mod sentinel;
pub mod server;
pub mod slot;
pub mod sse;
pub mod tune;
pub mod vram;

pub use client::{ChatRequest, ChatStream, FinishReason, LlamaClient, StreamEvent, Timings};
pub use gguf::{GgufFile, ModelFacts};
pub use manager::{EngineManager, EngineState};
pub use pidfile::EngineRecord;
pub use slot::{EngineSlot, SlotGuard};
pub use vram::{KvType, PlanOpts, VramPlan, VramPlanner};

pub type ContextId = u64;
