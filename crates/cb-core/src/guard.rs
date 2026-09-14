//! Runaway-loop protection.

use crate::message::ToolCall;
use std::collections::VecDeque;

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum GuardError {
    #[error("loop detected: the same tool call `{0}` was issued {1} times")]
    RepeatedCall(String, usize),
    #[error("loop detected: the same error repeated {0} times")]
    RepeatedError(usize),
    #[error("loop detected: the assistant produced the exact same response {0} times")]
    RepeatedOutput(usize),
    #[error("maximum number of turns ({0}) reached")]
    MaxTurns(u32),
}

pub struct LoopGuard {
    pub max_turns: u32,
    pub turn: u32,
    recent_calls: VecDeque<[u8; 32]>,
    recent_errors: VecDeque<[u8; 32]>,
    recent_texts: VecDeque<[u8; 32]>,
    window: usize,
    repeat_limit: usize,
}

impl LoopGuard {
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns, turn: 0, recent_calls: VecDeque::new(), recent_errors: VecDeque::new(), recent_texts: VecDeque::new(), window: 8, repeat_limit: 3 }
    }

    pub fn observe_calls(&mut self, calls: &[ToolCall]) -> Result<(), GuardError> {
        for c in calls {
            let fp = c.fingerprint();
            let count = self.recent_calls.iter().filter(|f| **f == fp).count() + 1;
            self.recent_calls.push_back(fp);
            while self.recent_calls.len() > self.window { self.recent_calls.pop_front(); }
            if count >= self.repeat_limit {
                return Err(GuardError::RepeatedCall(c.name.clone(), count));
            }
        }
        Ok(())
    }

    pub fn observe_error(&mut self, tool: &str, text: &str) -> Result<(), GuardError> {
        // Normalize: first 200 chars of the error, digits stripped (line numbers vary).
        let norm: String = text.chars().take(200).filter(|c| !c.is_ascii_digit()).collect();
        let fp = *blake3::hash(format!("{tool}\0{norm}").as_bytes()).as_bytes();
        let count = self.recent_errors.iter().filter(|f| **f == fp).count() + 1;
        self.recent_errors.push_back(fp);
        while self.recent_errors.len() > self.window { self.recent_errors.pop_front(); }
        if count >= self.repeat_limit { return Err(GuardError::RepeatedError(count)); }
        Ok(())
    }

    pub fn observe_text(&mut self, text: &str) -> Result<(), GuardError> {
        let trimmed = text.trim();
        if trimmed.is_empty() { return Ok(()); }
        let fp = *blake3::hash(trimmed.as_bytes()).as_bytes();
        let count = self.recent_texts.iter().filter(|f| **f == fp).count() + 1;
        self.recent_texts.push_back(fp);
        while self.recent_texts.len() > self.window { self.recent_texts.pop_front(); }
        if count >= self.repeat_limit { return Err(GuardError::RepeatedOutput(count)); }
        Ok(())
    }

    pub fn next_turn(&mut self) -> Result<(), GuardError> {
        self.turn += 1;
        if self.max_turns > 0 && self.turn >= self.max_turns { return Err(GuardError::MaxTurns(self.max_turns)); }
        Ok(())
    }

    /// Called when the user sends a new message: repetition history and turn counter reset for the new request.
    pub fn new_user_message(&mut self) {
        self.turn = 0;
        self.recent_calls.clear();
        self.recent_errors.clear();
        self.recent_texts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_repeat() {
        let mut g = LoopGuard::new(10);
        let c = ToolCall { id: "1".into(), name: "grep".into(), arguments: json!({"pattern":"x"}) };
        assert!(g.observe_calls(std::slice::from_ref(&c)).is_ok());
        assert!(g.observe_calls(std::slice::from_ref(&c)).is_ok());
        assert!(matches!(g.observe_calls(std::slice::from_ref(&c)), Err(GuardError::RepeatedCall(_, 3))));
        // Subsequent repeat properly advances count to 4:
        assert!(matches!(g.observe_calls(&[c]), Err(GuardError::RepeatedCall(_, 4))));
    }

    #[test]
    fn detects_repeated_output() {
        let mut g = LoopGuard::new(10);
        assert!(g.observe_text("I will check the file now.").is_ok());
        assert!(g.observe_text("I will check the file now.").is_ok());
        assert!(matches!(g.observe_text("I will check the file now."), Err(GuardError::RepeatedOutput(3))));
    }

    #[test]
    fn detects_repeated_errors() {
        let mut g = LoopGuard::new(10);
        assert!(g.observe_error("read_file", "file not found: foo.txt").is_ok());
        assert!(g.observe_error("read_file", "file not found: foo.txt").is_ok());
        assert!(matches!(g.observe_error("read_file", "file not found: foo.txt"), Err(GuardError::RepeatedError(3))));
    }
}
