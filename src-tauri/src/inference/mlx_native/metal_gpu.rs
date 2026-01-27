use super::metal_stubs::*;

/// Metal GPU device abstraction for Apple Silicon
pub struct MetalGPU {
    device: *mut std::ffi::c_void,
    command_queue: *mut std::ffi::c_void,
    library: *mut std::ffi::c_void,
}

impl MetalGPU {
    /// Create new Metal GPU device
    pub fn new() -> Result<Self, String> {
        unsafe {
            let device = metal_create_default_device();
            if device.is_null() {
                return Err("Failed to create Metal device".to_string());
            }

            let command_queue = metal_create_command_queue(device);
            if command_queue.is_null() {
                return Err("Failed to create command queue".to_string());
            }

            let library = metal_create_library(device);
            if library.is_null() {
                return Err("Failed to create Metal library".to_string());
            }

            Ok(MetalGPU {
                device,
                command_queue,
                library,
            })
        }
    }

    /// Allocate GPU buffer
    pub fn create_buffer(&self, size: usize) -> Result<*mut std::ffi::c_void, String> {
        unsafe {
            let buffer = metal_create_buffer(self.device, size);
            if buffer.is_null() {
                return Err(format!("Failed to allocate {} bytes on GPU", size));
            }
            Ok(buffer)
        }
    }

    /// Release GPU buffer
    pub fn release_buffer(&self, buffer: *mut std::ffi::c_void) {
        unsafe {
            metal_release_buffer(buffer);
        }
    }

    /// Copy data to GPU buffer
    pub fn copy_to_gpu(
        &self,
        gpu_buffer: *mut std::ffi::c_void,
        cpu_data: &[f32],
    ) -> Result<(), String> {
        unsafe {
            metal_copy_to_gpu(
                gpu_buffer,
                cpu_data.as_ptr() as *const std::ffi::c_void,
                cpu_data.len() * 4,
            );
            Ok(())
        }
    }

    /// Copy data from GPU buffer
    pub fn copy_from_gpu(
        &self,
        gpu_buffer: *mut std::ffi::c_void,
        cpu_data: &mut [f32],
    ) -> Result<(), String> {
        unsafe {
            metal_copy_from_gpu(
                gpu_buffer,
                cpu_data.as_mut_ptr() as *mut std::ffi::c_void,
                cpu_data.len() * 4,
            );
            Ok(())
        }
    }

    /// Create Metal command buffer for recording commands
    pub fn create_command_buffer(&self) -> Result<*mut std::ffi::c_void, String> {
        unsafe {
            let cmd_buffer = metal_create_command_buffer(self.command_queue);
            if cmd_buffer.is_null() {
                return Err("Failed to create command buffer".to_string());
            }
            Ok(cmd_buffer)
        }
    }

    /// Submit command buffer to GPU
    pub fn submit_commands(&self, cmd_buffer: *mut std::ffi::c_void) -> Result<(), String> {
        unsafe {
            metal_commit_command_buffer(cmd_buffer);
            Ok(())
        }
    }

    /// Wait for GPU execution to complete
    pub fn wait_completion(&self, cmd_buffer: *mut std::ffi::c_void) -> Result<(), String> {
        unsafe {
            metal_wait_command_buffer(cmd_buffer);
            Ok(())
        }
    }

    /// Get Metal device
    pub fn device(&self) -> *mut std::ffi::c_void {
        self.device
    }

    /// Get command queue
    pub fn command_queue(&self) -> *mut std::ffi::c_void {
        self.command_queue
    }

    /// Get Metal library for kernels
    pub fn library(&self) -> *mut std::ffi::c_void {
        self.library
    }

    /// Check if Metal is available on this system
    pub fn is_available() -> bool {
        unsafe { metal_is_available() }
    }
}

impl Drop for MetalGPU {
    fn drop(&mut self) {
        unsafe {
            if !self.library.is_null() {
                metal_release_library(self.library);
            }
            if !self.command_queue.is_null() {
                metal_release_command_queue(self.command_queue);
            }
            if !self.device.is_null() {
                metal_release_device(self.device);
            }
        }
    }
}
