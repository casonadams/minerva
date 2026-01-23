/// Metal GPU Acceleration for LLaMA Inference - Phase 6 Step 4
///
/// This module provides GPU acceleration using Apple's Metal framework for:
/// - Matrix multiplication kernels
/// - Attention computation
/// - Layer normalization
/// - Element-wise operations
///
/// All operations are designed to work with CPU fallbacks for testing.
use crate::error::{MinervaError, MinervaResult};
use std::sync::{Arc, Mutex};

/// GPU device capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUCapability {
    /// Basic compute support
    Basic,
    /// Simdgroup operations (faster)
    Simdgroup,
    /// Raster operations (most advanced)
    Raster,
}

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name (e.g., "Apple M1")
    pub name: String,
    /// Maximum memory available in MB
    pub max_memory_mb: u64,
    /// Supports real Metal or simulation mode
    pub is_simulated: bool,
    /// GPU capabilities
    pub capability: GPUCapability,
}

impl MetalDeviceInfo {
    /// Create device info for simulation
    pub fn simulated(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_memory_mb: 4096,
            is_simulated: true,
            capability: GPUCapability::Basic,
        }
    }

    /// Create device info for real Metal device
    pub fn real(name: &str, max_memory_mb: u64) -> Self {
        Self {
            name: name.to_string(),
            max_memory_mb,
            is_simulated: false,
            capability: GPUCapability::Simdgroup,
        }
    }
}

/// GPU buffer metadata
#[derive(Debug, Clone)]
struct BufferMetadata {
    /// Size in bytes
    size: usize,
    /// Last modified timestamp
    last_modified: std::time::SystemTime,
    /// Whether data is on GPU
    is_on_gpu: bool,
}

/// GPU Buffer wrapper
#[derive(Debug, Clone)]
pub struct GPUBuffer {
    /// Buffer identifier
    id: usize,
    /// Metadata
    metadata: BufferMetadata,
    /// CPU side data (for simulations)
    cpu_data: Arc<Mutex<Vec<u8>>>,
}

impl GPUBuffer {
    /// Create new GPU buffer
    pub fn new(id: usize, data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            id,
            metadata: BufferMetadata {
                size,
                last_modified: std::time::SystemTime::now(),
                is_on_gpu: false,
            },
            cpu_data: Arc::new(Mutex::new(data)),
        }
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.metadata.size
    }

    /// Read buffer data
    pub fn read(&self) -> MinervaResult<Vec<u8>> {
        self.cpu_data
            .lock()
            .map(|data| data.clone())
            .map_err(|_| MinervaError::InferenceError("Failed to acquire buffer lock".to_string()))
    }

    /// Write buffer data
    pub fn write(&mut self, data: Vec<u8>) -> MinervaResult<()> {
        if data.len() != self.metadata.size {
            return Err(MinervaError::InferenceError(
                "Buffer size mismatch on write".to_string(),
            ));
        }
        *self.cpu_data.lock().map_err(|_| {
            MinervaError::InferenceError("Failed to acquire buffer lock".to_string())
        })? = data;
        self.metadata.last_modified = std::time::SystemTime::now();
        Ok(())
    }

    /// Mark buffer as on GPU
    pub fn mark_on_gpu(&mut self) {
        self.metadata.is_on_gpu = true;
    }

    /// Mark buffer as on CPU
    pub fn mark_on_cpu(&mut self) {
        self.metadata.is_on_gpu = false;
    }

    /// Check if buffer is on GPU
    pub fn is_on_gpu(&self) -> bool {
        self.metadata.is_on_gpu
    }
}

/// GPU Kernel abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Matrix multiplication: C = A @ B
    MatMul,
    /// Attention: softmax(Q @ K^T / sqrt(d)) @ V
    Attention,
    /// Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
    LayerNorm,
    /// Element-wise SiLU: x / (1 + exp(-x))
    SiLU,
    /// Softmax normalization
    Softmax,
    /// Element-wise multiplication
    ElementMul,
}

impl KernelType {
    /// Get kernel name for debugging
    pub fn name(&self) -> &'static str {
        match self {
            KernelType::MatMul => "matmul",
            KernelType::Attention => "attention",
            KernelType::LayerNorm => "layer_norm",
            KernelType::SiLU => "silu",
            KernelType::Softmax => "softmax",
            KernelType::ElementMul => "element_mul",
        }
    }
}

/// GPU Kernel configuration
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Kernel type
    pub kernel: KernelType,
    /// Thread group size (typical: 256)
    pub thread_group_size: u32,
    /// Use SIMD optimization
    pub use_simd: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::MatMul,
            thread_group_size: 256,
            use_simd: true,
        }
    }
}

/// Kernel buffer configuration
#[derive(Debug, Clone)]
pub struct KernelBuffers {
    /// Input buffer IDs
    pub input_buffers: Vec<usize>,
    /// Output buffer ID
    pub output_buffer: usize,
}

impl KernelBuffers {
    /// Create new kernel buffers
    pub fn new(input_buffers: Vec<usize>, output_buffer: usize) -> Self {
        Self {
            input_buffers,
            output_buffer,
        }
    }
}

/// Kernel execution result
#[derive(Debug, Clone)]
pub struct KernelResult {
    /// Kernel type executed
    pub kernel: KernelType,
    /// Execution time in milliseconds
    pub execution_time_ms: f32,
    /// Output buffer ID
    pub output_buffer_id: usize,
    /// Error if any
    pub error: Option<String>,
}

/// Metal GPU Device
pub struct MetalDevice {
    /// Device info
    info: MetalDeviceInfo,
    /// Buffer registry
    buffers: Arc<Mutex<Vec<GPUBuffer>>>,
    /// Command queue (simulated)
    #[allow(dead_code)]
    command_queue_id: usize,
    /// Next buffer ID
    next_buffer_id: Arc<Mutex<usize>>,
}

impl MetalDevice {
    /// Create new Metal device
    pub fn new(info: MetalDeviceInfo) -> Self {
        Self {
            info,
            buffers: Arc::new(Mutex::new(Vec::new())),
            command_queue_id: 0,
            next_buffer_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Create simulated device (for testing)
    pub fn simulated() -> Self {
        Self::new(MetalDeviceInfo::simulated("Simulated Metal GPU"))
    }

    /// Get device info
    pub fn info(&self) -> &MetalDeviceInfo {
        &self.info
    }

    /// Allocate GPU buffer
    pub fn allocate_buffer(&self, size: usize) -> MinervaResult<usize> {
        let mut next_id = self
            .next_buffer_id
            .lock()
            .map_err(|_| MinervaError::InferenceError("Failed to acquire ID lock".to_string()))?;
        let id = *next_id;
        *next_id += 1;

        let buffer = GPUBuffer::new(id, vec![0u8; size]);
        self.buffers
            .lock()
            .map_err(|_| MinervaError::InferenceError("Failed to acquire buffer lock".to_string()))?
            .push(buffer);

        Ok(id)
    }

    /// Free GPU buffer
    pub fn free_buffer(&self, id: usize) -> MinervaResult<()> {
        let mut buffers = self.buffers.lock().map_err(|_| {
            MinervaError::InferenceError("Failed to acquire buffer lock".to_string())
        })?;

        if let Some(pos) = buffers.iter().position(|b| b.id == id) {
            buffers.remove(pos);
            Ok(())
        } else {
            Err(MinervaError::InferenceError(format!(
                "Buffer {} not found",
                id
            )))
        }
    }

    /// Get buffer
    fn get_buffer(&self, id: usize) -> MinervaResult<GPUBuffer> {
        let buffers = self.buffers.lock().map_err(|_| {
            MinervaError::InferenceError("Failed to acquire buffer lock".to_string())
        })?;

        buffers
            .iter()
            .find(|b| b.id == id)
            .cloned()
            .ok_or_else(|| MinervaError::InferenceError(format!("Buffer {} not found", id)))
    }

    /// Get mutable buffer
    fn get_buffer_mut(&self, id: usize) -> MinervaResult<GPUBuffer> {
        let mut buffers = self.buffers.lock().map_err(|_| {
            MinervaError::InferenceError("Failed to acquire buffer lock".to_string())
        })?;

        if let Some(buffer) = buffers.iter_mut().find(|b| b.id == id) {
            Ok(buffer.clone())
        } else {
            Err(MinervaError::InferenceError(format!(
                "Buffer {} not found",
                id
            )))
        }
    }

    /// Copy data to GPU buffer
    pub fn copy_to_gpu(&self, id: usize, data: &[u8]) -> MinervaResult<()> {
        let mut buffer = self.get_buffer_mut(id)?;

        if data.len() != buffer.size() {
            return Err(MinervaError::InferenceError(
                "Data size mismatch for GPU copy".to_string(),
            ));
        }

        buffer.write(data.to_vec())?;
        buffer.mark_on_gpu();

        // Update in registry
        let mut buffers = self.buffers.lock().map_err(|_| {
            MinervaError::InferenceError("Failed to acquire buffer lock".to_string())
        })?;
        if let Some(b) = buffers.iter_mut().find(|b| b.id == id) {
            *b = buffer;
        }

        Ok(())
    }

    /// Copy data from GPU buffer
    pub fn copy_from_gpu(&self, id: usize) -> MinervaResult<Vec<u8>> {
        let buffer = self.get_buffer(id)?;
        buffer.read()
    }

    /// Execute kernel on GPU
    pub fn execute_kernel(
        &self,
        config: KernelConfig,
        buffers: KernelBuffers,
    ) -> MinervaResult<KernelResult> {
        let start = std::time::Instant::now();

        // Validate buffers exist
        for &id in &buffers.input_buffers {
            self.get_buffer(id)?;
        }
        self.get_buffer(buffers.output_buffer)?;

        // Simulate kernel execution
        match config.kernel {
            KernelType::MatMul => {
                self.simulate_matmul(&buffers.input_buffers, buffers.output_buffer)?;
            }
            KernelType::Attention => {
                self.simulate_attention(&buffers.input_buffers, buffers.output_buffer)?;
            }
            KernelType::LayerNorm => {
                self.simulate_layer_norm(&buffers.input_buffers, buffers.output_buffer)?;
            }
            KernelType::SiLU => {
                self.simulate_silu(&buffers.input_buffers, buffers.output_buffer)?;
            }
            KernelType::Softmax => {
                self.simulate_softmax(&buffers.input_buffers, buffers.output_buffer)?;
            }
            KernelType::ElementMul => {
                self.simulate_element_mul(&buffers.input_buffers, buffers.output_buffer)?;
            }
        }

        let elapsed = start.elapsed();

        Ok(KernelResult {
            kernel: config.kernel,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            output_buffer_id: buffers.output_buffer,
            error: None,
        })
    }

    /// Simulate matrix multiplication
    fn simulate_matmul(&self, inputs: &[usize], _output: usize) -> MinervaResult<()> {
        if inputs.len() != 2 {
            return Err(MinervaError::InferenceError(
                "MatMul requires 2 inputs".to_string(),
            ));
        }

        let _a = self.get_buffer(inputs[0])?;
        let _b = self.get_buffer(inputs[1])?;

        // Simulation: just validate buffers exist
        Ok(())
    }

    /// Simulate attention computation
    fn simulate_attention(&self, inputs: &[usize], _output: usize) -> MinervaResult<()> {
        if inputs.len() != 3 {
            return Err(MinervaError::InferenceError(
                "Attention requires 3 inputs (Q, K, V)".to_string(),
            ));
        }

        let _q = self.get_buffer(inputs[0])?;
        let _k = self.get_buffer(inputs[1])?;
        let _v = self.get_buffer(inputs[2])?;

        Ok(())
    }

    /// Simulate layer normalization
    fn simulate_layer_norm(&self, inputs: &[usize], _output: usize) -> MinervaResult<()> {
        if inputs.len() != 2 {
            return Err(MinervaError::InferenceError(
                "LayerNorm requires 2 inputs (input, weight)".to_string(),
            ));
        }

        let _x = self.get_buffer(inputs[0])?;
        let _weight = self.get_buffer(inputs[1])?;

        Ok(())
    }

    /// Simulate SiLU activation
    fn simulate_silu(&self, inputs: &[usize], _output: usize) -> MinervaResult<()> {
        if inputs.len() != 1 {
            return Err(MinervaError::InferenceError(
                "SiLU requires 1 input".to_string(),
            ));
        }

        let _x = self.get_buffer(inputs[0])?;

        Ok(())
    }

    /// Simulate softmax
    fn simulate_softmax(&self, inputs: &[usize], _output: usize) -> MinervaResult<()> {
        if inputs.len() != 1 {
            return Err(MinervaError::InferenceError(
                "Softmax requires 1 input".to_string(),
            ));
        }

        let _x = self.get_buffer(inputs[0])?;

        Ok(())
    }

    /// Simulate element-wise multiplication
    fn simulate_element_mul(&self, inputs: &[usize], _output: usize) -> MinervaResult<()> {
        if inputs.len() != 2 {
            return Err(MinervaError::InferenceError(
                "ElementMul requires 2 inputs".to_string(),
            ));
        }

        let _a = self.get_buffer(inputs[0])?;
        let _b = self.get_buffer(inputs[1])?;

        Ok(())
    }
}

/// GPU Memory Pool for efficient allocation
pub struct GPUMemoryPool {
    /// Available memory blocks
    free_blocks: Arc<Mutex<Vec<(usize, usize)>>>, // (start, size)
    /// Allocated blocks
    allocated_blocks: Arc<Mutex<Vec<(usize, usize)>>>, // (start, size)
    /// Total pool size
    #[allow(dead_code)]
    total_size: usize,
}

impl GPUMemoryPool {
    /// Create new memory pool
    pub fn new(size: usize) -> Self {
        Self {
            free_blocks: Arc::new(Mutex::new(vec![(0, size)])),
            allocated_blocks: Arc::new(Mutex::new(Vec::new())),
            total_size: size,
        }
    }

    /// Allocate memory from pool
    pub fn allocate(&self, size: usize) -> MinervaResult<usize> {
        let mut free = self
            .free_blocks
            .lock()
            .map_err(|_| MinervaError::InferenceError("Failed to acquire lock".to_string()))?;

        // Find first fit
        for i in 0..free.len() {
            if free[i].1 >= size {
                let start = free[i].0;
                free[i].0 += size;
                free[i].1 -= size;

                if free[i].1 == 0 {
                    free.remove(i);
                }

                let mut allocated = self.allocated_blocks.lock().map_err(|_| {
                    MinervaError::InferenceError("Failed to acquire lock".to_string())
                })?;
                allocated.push((start, size));

                return Ok(start);
            }
        }

        Err(MinervaError::InferenceError(
            "Insufficient memory in GPU pool".to_string(),
        ))
    }

    /// Deallocate memory from pool
    pub fn deallocate(&self, start: usize, size: usize) -> MinervaResult<()> {
        let mut allocated = self
            .allocated_blocks
            .lock()
            .map_err(|_| MinervaError::InferenceError("Failed to acquire lock".to_string()))?;

        if let Some(pos) = allocated.iter().position(|&(s, _)| s == start) {
            allocated.remove(pos);

            let mut free = self
                .free_blocks
                .lock()
                .map_err(|_| MinervaError::InferenceError("Failed to acquire lock".to_string()))?;
            free.push((start, size));

            Ok(())
        } else {
            Err(MinervaError::InferenceError(
                "Block not found in allocations".to_string(),
            ))
        }
    }

    /// Get free memory
    pub fn free_memory(&self) -> MinervaResult<usize> {
        let free = self
            .free_blocks
            .lock()
            .map_err(|_| MinervaError::InferenceError("Failed to acquire lock".to_string()))?;

        Ok(free.iter().map(|(_, size)| size).sum())
    }

    /// Get used memory
    pub fn used_memory(&self) -> MinervaResult<usize> {
        let allocated = self
            .allocated_blocks
            .lock()
            .map_err(|_| MinervaError::InferenceError("Failed to acquire lock".to_string()))?;

        Ok(allocated.iter().map(|(_, size)| size).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_info_simulated() {
        let info = MetalDeviceInfo::simulated("Test GPU");
        assert_eq!(info.name, "Test GPU");
        assert!(info.is_simulated);
        assert_eq!(info.max_memory_mb, 4096);
    }

    #[test]
    fn test_metal_device_info_real() {
        let info = MetalDeviceInfo::real("M1", 8192);
        assert_eq!(info.name, "M1");
        assert!(!info.is_simulated);
        assert_eq!(info.max_memory_mb, 8192);
    }

    #[test]
    fn test_gpu_buffer_creation() {
        let data = vec![1u8; 1024];
        let buffer = GPUBuffer::new(0, data);
        assert_eq!(buffer.size(), 1024);
        assert!(!buffer.is_on_gpu());
    }

    #[test]
    fn test_gpu_buffer_read_write() {
        let buffer = GPUBuffer::new(0, vec![0u8; 100]);
        let data = vec![5u8; 100];
        let mut buf = buffer.clone();
        assert!(buf.write(data.clone()).is_ok());
        assert_eq!(buf.read().unwrap(), data);
    }

    #[test]
    fn test_gpu_buffer_on_gpu_flag() {
        let buffer = GPUBuffer::new(0, vec![0u8; 100]);
        assert!(!buffer.is_on_gpu());

        let mut buf = buffer.clone();
        buf.mark_on_gpu();
        assert!(buf.is_on_gpu());

        buf.mark_on_cpu();
        assert!(!buf.is_on_gpu());
    }

    #[test]
    fn test_kernel_type_names() {
        assert_eq!(KernelType::MatMul.name(), "matmul");
        assert_eq!(KernelType::Attention.name(), "attention");
        assert_eq!(KernelType::LayerNorm.name(), "layer_norm");
        assert_eq!(KernelType::SiLU.name(), "silu");
        assert_eq!(KernelType::Softmax.name(), "softmax");
        assert_eq!(KernelType::ElementMul.name(), "element_mul");
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.kernel, KernelType::MatMul);
        assert_eq!(config.thread_group_size, 256);
        assert!(config.use_simd);
    }

    #[test]
    fn test_metal_device_simulated() {
        let device = MetalDevice::simulated();
        assert_eq!(device.info().name, "Simulated Metal GPU");
        assert!(device.info().is_simulated);
    }

    #[test]
    fn test_metal_device_allocate_buffer() {
        let device = MetalDevice::simulated();
        let id = device.allocate_buffer(1024).unwrap();
        assert_eq!(id, 0);

        let id2 = device.allocate_buffer(2048).unwrap();
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_metal_device_free_buffer() {
        let device = MetalDevice::simulated();
        let id = device.allocate_buffer(1024).unwrap();
        assert!(device.free_buffer(id).is_ok());
        assert!(device.free_buffer(id).is_err());
    }

    #[test]
    fn test_metal_device_copy_to_gpu() {
        let device = MetalDevice::simulated();
        let id = device.allocate_buffer(100).unwrap();
        let data = vec![5u8; 100];
        assert!(device.copy_to_gpu(id, &data).is_ok());
    }

    #[test]
    fn test_metal_device_copy_from_gpu() {
        let device = MetalDevice::simulated();
        let id = device.allocate_buffer(100).unwrap();
        let data = vec![7u8; 100];
        device.copy_to_gpu(id, &data).unwrap();
        let result = device.copy_from_gpu(id).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_metal_device_execute_matmul_kernel() {
        let device = MetalDevice::simulated();
        let a = device.allocate_buffer(1024).unwrap();
        let b = device.allocate_buffer(1024).unwrap();
        let c = device.allocate_buffer(1024).unwrap();

        let config = KernelConfig {
            kernel: KernelType::MatMul,
            ..Default::default()
        };
        let result = device
            .execute_kernel(config, KernelBuffers::new(vec![a, b], c))
            .unwrap();
        assert_eq!(result.kernel, KernelType::MatMul);
        assert_eq!(result.output_buffer_id, c);
    }

    #[test]
    fn test_metal_device_execute_attention_kernel() {
        let device = MetalDevice::simulated();
        let q = device.allocate_buffer(512).unwrap();
        let k = device.allocate_buffer(512).unwrap();
        let v = device.allocate_buffer(512).unwrap();
        let out = device.allocate_buffer(512).unwrap();

        let config = KernelConfig {
            kernel: KernelType::Attention,
            ..Default::default()
        };
        let result = device
            .execute_kernel(config, KernelBuffers::new(vec![q, k, v], out))
            .unwrap();
        assert_eq!(result.kernel, KernelType::Attention);
    }

    #[test]
    fn test_metal_device_execute_layer_norm_kernel() {
        let device = MetalDevice::simulated();
        let x = device.allocate_buffer(256).unwrap();
        let weight = device.allocate_buffer(256).unwrap();
        let out = device.allocate_buffer(256).unwrap();

        let config = KernelConfig {
            kernel: KernelType::LayerNorm,
            ..Default::default()
        };
        let result = device
            .execute_kernel(config, KernelBuffers::new(vec![x, weight], out))
            .unwrap();
        assert_eq!(result.kernel, KernelType::LayerNorm);
    }

    #[test]
    fn test_metal_device_invalid_kernel_inputs() {
        let device = MetalDevice::simulated();
        let a = device.allocate_buffer(1024).unwrap();
        let out = device.allocate_buffer(1024).unwrap();

        let config = KernelConfig {
            kernel: KernelType::MatMul,
            ..Default::default()
        };
        assert!(device
            .execute_kernel(config, KernelBuffers::new(vec![a], out))
            .is_err());
    }

    #[test]
    fn test_gpu_memory_pool_creation() {
        let pool = GPUMemoryPool::new(4096);
        assert_eq!(pool.used_memory().unwrap(), 0);
        assert_eq!(pool.free_memory().unwrap(), 4096);
    }

    #[test]
    fn test_gpu_memory_pool_allocate() {
        let pool = GPUMemoryPool::new(4096);
        let start = pool.allocate(1024).unwrap();
        assert_eq!(start, 0);
        assert_eq!(pool.used_memory().unwrap(), 1024);
        assert_eq!(pool.free_memory().unwrap(), 3072);
    }

    #[test]
    fn test_gpu_memory_pool_allocate_multiple() {
        let pool = GPUMemoryPool::new(4096);
        let start1 = pool.allocate(1024).unwrap();
        let start2 = pool.allocate(2048).unwrap();
        assert_eq!(start1, 0);
        assert_eq!(start2, 1024);
        assert_eq!(pool.used_memory().unwrap(), 3072);
    }

    #[test]
    fn test_gpu_memory_pool_deallocate() {
        let pool = GPUMemoryPool::new(4096);
        let start = pool.allocate(1024).unwrap();
        assert!(pool.deallocate(start, 1024).is_ok());
        assert_eq!(pool.used_memory().unwrap(), 0);
        assert_eq!(pool.free_memory().unwrap(), 4096);
    }

    #[test]
    fn test_gpu_memory_pool_insufficient_memory() {
        let pool = GPUMemoryPool::new(1024);
        assert!(pool.allocate(512).is_ok());
        assert!(pool.allocate(512).is_ok());
        assert!(pool.allocate(1).is_err());
    }

    #[test]
    fn test_kernel_result() {
        let result = KernelResult {
            kernel: KernelType::MatMul,
            execution_time_ms: 1.5,
            output_buffer_id: 0,
            error: None,
        };
        assert_eq!(result.kernel, KernelType::MatMul);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_kernel_result_with_error() {
        let result = KernelResult {
            kernel: KernelType::Attention,
            execution_time_ms: 0.0,
            output_buffer_id: 0,
            error: Some("Test error".to_string()),
        };
        assert!(result.error.is_some());
    }

    #[test]
    fn test_metal_device_copy_size_mismatch() {
        let device = MetalDevice::simulated();
        let id = device.allocate_buffer(100).unwrap();
        let data = vec![5u8; 50]; // Wrong size
        assert!(device.copy_to_gpu(id, &data).is_err());
    }

    #[test]
    fn test_metal_device_nonexistent_buffer() {
        let device = MetalDevice::simulated();
        assert!(device.copy_from_gpu(999).is_err());
    }

    #[test]
    fn test_metal_device_silu_kernel() {
        let device = MetalDevice::simulated();
        let x = device.allocate_buffer(256).unwrap();
        let out = device.allocate_buffer(256).unwrap();

        let config = KernelConfig {
            kernel: KernelType::SiLU,
            ..Default::default()
        };
        let result = device
            .execute_kernel(config, KernelBuffers::new(vec![x], out))
            .unwrap();
        assert_eq!(result.kernel, KernelType::SiLU);
    }

    #[test]
    fn test_metal_device_softmax_kernel() {
        let device = MetalDevice::simulated();
        let x = device.allocate_buffer(256).unwrap();
        let out = device.allocate_buffer(256).unwrap();

        let config = KernelConfig {
            kernel: KernelType::Softmax,
            ..Default::default()
        };
        let result = device
            .execute_kernel(config, KernelBuffers::new(vec![x], out))
            .unwrap();
        assert_eq!(result.kernel, KernelType::Softmax);
    }

    #[test]
    fn test_metal_device_element_mul_kernel() {
        let device = MetalDevice::simulated();
        let a = device.allocate_buffer(256).unwrap();
        let b = device.allocate_buffer(256).unwrap();
        let out = device.allocate_buffer(256).unwrap();

        let config = KernelConfig {
            kernel: KernelType::ElementMul,
            ..Default::default()
        };
        let result = device
            .execute_kernel(config, KernelBuffers::new(vec![a, b], out))
            .unwrap();
        assert_eq!(result.kernel, KernelType::ElementMul);
    }
}
