//! Hardware, GPU, and RAM discovery across Windows, macOS (Metal), and Linux.

pub mod types;
pub use types::*;

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use windows::{free_mib, query, system_ram};

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub use macos::{free_mib, query, system_ram};

#[cfg(all(not(windows), not(target_os = "macos")))]
pub mod unix;
#[cfg(all(not(windows), not(target_os = "macos")))]
pub use unix::{free_mib, query, system_ram};
