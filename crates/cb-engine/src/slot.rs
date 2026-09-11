//! Single-slot serialization. MTP speculative decoding requires `--parallel 1`, so every
//! generation (main agent, subagents, summaries) must queue here. Tracks which context last
//! owned the slot so callers can expect/record a prompt-cache switch.

use crate::ContextId;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug)]
pub struct EngineSlot {
    sem: Arc<Semaphore>,
    owner: Mutex<Option<ContextId>>,
    switches: Mutex<u64>,
}

pub struct SlotGuard {
    _permit: OwnedSemaphorePermit,
    /// True if a different context owned the slot before us (prompt cache likely evicted).
    pub switched: bool,
    pub previous_owner: Option<ContextId>,
}

impl Default for EngineSlot {
    fn default() -> Self { Self::new() }
}

impl EngineSlot {
    pub fn new() -> Self { Self { sem: Arc::new(Semaphore::new(1)), owner: Mutex::new(None), switches: Mutex::new(0) } }

    pub async fn acquire(&self, ctx: ContextId) -> SlotGuard {
        let permit = self.sem.clone().acquire_owned().await.expect("slot semaphore closed");
        let mut owner = self.owner.lock();
        let previous_owner = *owner;
        let switched = previous_owner.is_some_and(|p| p != ctx);
        if switched { *self.switches.lock() += 1; }
        *owner = Some(ctx);
        SlotGuard { _permit: permit, switched, previous_owner }
    }

    pub fn try_acquire(&self, ctx: ContextId) -> Option<SlotGuard> {
        let permit = self.sem.clone().try_acquire_owned().ok()?;
        let mut owner = self.owner.lock();
        let previous_owner = *owner;
        let switched = previous_owner.is_some_and(|p| p != ctx);
        if switched { *self.switches.lock() += 1; }
        *owner = Some(ctx);
        Some(SlotGuard { _permit: permit, switched, previous_owner })
    }

    pub fn is_busy(&self) -> bool { self.sem.available_permits() == 0 }
    pub fn current_owner(&self) -> Option<ContextId> { *self.owner.lock() }
    pub fn switch_count(&self) -> u64 { *self.switches.lock() }
    /// Forget the owner (e.g. after a server restart: the cache is gone anyway).
    pub fn reset_owner(&self) { *self.owner.lock() = None; }
}
