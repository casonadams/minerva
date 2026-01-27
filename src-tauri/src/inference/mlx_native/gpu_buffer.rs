use super::metal_gpu::MetalGPU;
use std::sync::Arc;

/// GPU buffer with metadata for pooling
#[derive(Clone)]
pub struct GPUBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
    gpu: Arc<MetalGPU>,
}

impl GPUBuffer {
    pub fn new(ptr: *mut std::ffi::c_void, size: usize, gpu: Arc<MetalGPU>) -> Self {
        GPUBuffer { ptr, size, gpu }
    }

    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for GPUBuffer {
    fn drop(&mut self) {
        self.gpu.release_buffer(self.ptr);
    }
}
