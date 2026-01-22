use crate::error::{MinervaError, MinervaResult};

/// GPU device type
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum GpuDevice {
    /// Metal (Apple Silicon / AMD)
    Metal,
    /// CUDA (NVIDIA)
    Cuda,
    /// CPU fallback
    Cpu,
}

/// GPU context for hardware acceleration
#[derive(Debug)]
#[allow(dead_code)]
pub struct GpuContext {
    device: GpuDevice,
    allocated_memory: usize,
    max_memory: usize,
}

#[allow(dead_code)]
impl GpuContext {
    /// Create GPU context with auto-detection
    pub fn new() -> MinervaResult<Self> {
        let device = Self::detect_device();

        let max_memory = match device {
            GpuDevice::Metal => {
                // Metal typically uses unified memory on Apple Silicon
                // Usually can allocate 50% of system RAM for GPU
                Self::estimate_memory(0.5)
            }
            GpuDevice::Cuda => {
                // CUDA VRAM detection would go here
                Self::estimate_memory(0.8)
            }
            GpuDevice::Cpu => Self::estimate_memory(0.25),
        };

        tracing::info!(
            "GPU Context initialized: device={:?}, max_memory={}MB",
            device,
            max_memory / 1_000_000
        );

        Ok(Self {
            device,
            allocated_memory: 0,
            max_memory,
        })
    }

    /// Get active device
    pub fn device(&self) -> GpuDevice {
        self.device
    }

    /// Allocate memory on GPU
    pub fn allocate(&mut self, size: usize) -> MinervaResult<()> {
        if self.allocated_memory + size > self.max_memory {
            return Err(MinervaError::OutOfMemory(format!(
                "GPU memory exceeded: {} + {} > {}",
                self.allocated_memory, size, self.max_memory
            )));
        }

        self.allocated_memory += size;
        tracing::debug!(
            "GPU memory allocated: {} bytes, total: {} bytes",
            size,
            self.allocated_memory
        );

        Ok(())
    }

    /// Free GPU memory
    pub fn deallocate(&mut self, size: usize) -> MinervaResult<()> {
        if size > self.allocated_memory {
            return Err(MinervaError::InferenceError(
                "Attempting to deallocate more than allocated".to_string(),
            ));
        }

        self.allocated_memory -= size;
        tracing::debug!(
            "GPU memory freed: {} bytes, remaining: {} bytes",
            size,
            self.allocated_memory
        );

        Ok(())
    }

    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.max_memory.saturating_sub(self.allocated_memory)
    }

    /// Get allocated memory
    pub fn allocated_memory(&self) -> usize {
        self.allocated_memory
    }

    /// Get max memory
    pub fn max_memory(&self) -> usize {
        self.max_memory
    }

    /// Detect available GPU device
    fn detect_device() -> GpuDevice {
        #[cfg(target_os = "macos")]
        {
            GpuDevice::Metal
        }
        #[cfg(target_os = "windows")]
        {
            // On Windows, prefer CUDA if available, else CPU
            if Self::has_cuda() {
                GpuDevice::Cuda
            } else {
                GpuDevice::Cpu
            }
        }
        #[cfg(target_os = "linux")]
        {
            // On Linux, prefer CUDA if available, else CPU
            if Self::has_cuda() {
                GpuDevice::Cuda
            } else {
                GpuDevice::Cpu
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            GpuDevice::Cpu
        }
    }

    /// Check if CUDA is available
    fn has_cuda() -> bool {
        // Simple check: look for CUDA libraries
        // Phase 3.5: Replace with actual CUDA detection
        std::env::var("CUDA_PATH").is_ok()
    }

    /// Estimate available memory based on ratio
    fn estimate_memory(ratio: f32) -> usize {
        // Use system memory info to estimate
        // For now, use reasonable defaults
        let system_memory = 16usize * 1024 * 1024 * 1024; // 16GB default
        (system_memory as f32 * ratio) as usize
    }
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            device: GpuDevice::Cpu,
            allocated_memory: 0,
            max_memory: 1024 * 1024 * 1024, // 1GB fallback
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new().unwrap_or_default();
        assert!(ctx.available_memory() > 0);
    }

    #[test]
    fn test_gpu_context_device_detection() {
        let ctx = GpuContext::new().unwrap_or_default();
        let device = ctx.device();
        assert!(matches!(
            device,
            GpuDevice::Metal | GpuDevice::Cuda | GpuDevice::Cpu
        ));
    }

    #[test]
    fn test_gpu_context_allocate() {
        let mut ctx = GpuContext::new().unwrap_or_default();
        let initial = ctx.allocated_memory();
        let size = 100 * 1024 * 1024; // 100MB

        assert!(ctx.allocate(size).is_ok());
        assert_eq!(ctx.allocated_memory(), initial + size);
    }

    #[test]
    fn test_gpu_context_deallocate() {
        let mut ctx = GpuContext::new().unwrap_or_default();
        let size = 100 * 1024 * 1024; // 100MB

        assert!(ctx.allocate(size).is_ok());
        assert!(ctx.deallocate(size).is_ok());
        assert_eq!(ctx.allocated_memory(), 0);
    }

    #[test]
    fn test_gpu_context_out_of_memory() {
        let mut ctx = GpuContext {
            device: GpuDevice::Cpu,
            allocated_memory: 100,
            max_memory: 200,
        };

        assert!(ctx.allocate(200).is_err());
    }

    #[test]
    fn test_gpu_context_available_memory() {
        let mut ctx = GpuContext {
            device: GpuDevice::Cpu,
            allocated_memory: 100,
            max_memory: 500,
        };

        assert_eq!(ctx.available_memory(), 400);

        assert!(ctx.allocate(100).is_ok());
        assert_eq!(ctx.available_memory(), 300);
    }
}
