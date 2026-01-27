use super::gpu_buffer::GPUBuffer;
use super::metal_gpu::MetalGPU;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU buffer pool for memory reuse
pub struct BufferPool {
    gpu: Arc<MetalGPU>,
    available: Arc<Mutex<HashMap<usize, Vec<*mut std::ffi::c_void>>>>,
    total_allocated: Arc<Mutex<usize>>,
    max_capacity: usize,
}

impl BufferPool {
    /// Create new buffer pool
    pub fn new(gpu: Arc<MetalGPU>, max_capacity: usize) -> Self {
        BufferPool {
            gpu,
            available: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            max_capacity,
        }
    }

    /// Allocate or reuse a buffer of specified size
    pub fn allocate(&self, size: usize) -> Result<GPUBuffer, String> {
        let mut available = self.available.lock().unwrap();
        let mut total = self.total_allocated.lock().unwrap();

        if let Some(buffers) = available.get_mut(&size) {
            if let Some(ptr) = buffers.pop() {
                return Ok(GPUBuffer::new(ptr, size, Arc::clone(&self.gpu)));
            }
        }

        if *total + size > self.max_capacity {
            self.evict_lru(&mut available, size)?;
        }

        let ptr = self.gpu.create_buffer(size)?;
        *total += size;

        Ok(GPUBuffer::new(ptr, size, Arc::clone(&self.gpu)))
    }

    /// Return buffer to pool for reuse
    pub fn release(&self, buffer: GPUBuffer) {
        let mut available = self.available.lock().unwrap();
        available
            .entry(buffer.size())
            .or_insert_with(Vec::new)
            .push(buffer.ptr());
    }

    /// Evict least recently used buffer
    fn evict_lru(
        &self,
        available: &mut HashMap<usize, Vec<*mut std::ffi::c_void>>,
        _needed: usize,
    ) -> Result<(), String> {
        if let Some(size) = available.keys().next().copied() {
            if let Some(mut buffers) = available.remove(&size) {
                while let Some(ptr) = buffers.pop() {
                    self.gpu.release_buffer(ptr);
                }
            }
            Ok(())
        } else {
            Err("No buffers to evict".to_string())
        }
    }

    /// Get current memory usage statistics
    pub fn statistics(&self) -> PoolStatistics {
        let available = self.available.lock().unwrap();
        let total = self.total_allocated.lock().unwrap();

        let available_count = available.values().map(|v| v.len()).sum();
        let available_bytes = available.keys().map(|&size| size).sum();

        PoolStatistics {
            total_allocated: *total,
            available_bytes,
            available_buffers: available_count,
            max_capacity: self.max_capacity,
        }
    }

    /// Clear all buffers in pool
    pub fn clear(&self) {
        let mut available = self.available.lock().unwrap();
        available.clear();
        *self.total_allocated.lock().unwrap() = 0;
    }
}

/// Statistics about buffer pool
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub total_allocated: usize,
    pub available_bytes: usize,
    pub available_buffers: usize,
    pub max_capacity: usize,
}
