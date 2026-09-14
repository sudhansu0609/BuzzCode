//! GPU and memory data types shared across platforms.

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct GpuMem {
    pub total_mib: u64,
    pub used_mib: u64,
    pub free_mib: u64,
}

impl GpuMem {
    pub fn total_bytes(&self) -> u64 { self.total_mib << 20 }
    pub fn free_bytes(&self) -> u64 { self.free_mib << 20 }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub driver: String,
    pub mem: GpuMem,
}
