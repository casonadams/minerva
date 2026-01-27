use super::metal_gpu::MetalGPU;
use super::metal_stubs::*;

/// Metal kernel operations
pub struct MetalKernels;

impl MetalKernels {
    /// Create Metal function from kernel name
    pub fn get_function(gpu: &MetalGPU, name: &str) -> Result<*mut std::ffi::c_void, String> {
        unsafe {
            let c_name =
                std::ffi::CString::new(name).map_err(|_| "Invalid kernel name".to_string())?;
            let func = metal_create_function(gpu.library(), c_name.as_ptr());
            if func.is_null() {
                return Err(format!("Failed to load kernel: {}", name));
            }
            Ok(func)
        }
    }

    /// Execute matmul kernel
    pub fn matmul(
        gpu: &MetalGPU,
        a: *mut std::ffi::c_void,
        b: *mut std::ffi::c_void,
        c: *mut std::ffi::c_void,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), String> {
        unsafe {
            let func = Self::get_function(gpu, "matmul_kernel")?;
            let cmd_buffer = gpu.create_command_buffer()?;
            metal_dispatch_matmul(cmd_buffer, func, a, b, c, m, n, k);
            gpu.submit_commands(cmd_buffer)?;
            Ok(())
        }
    }

    /// Execute add kernel
    pub fn add(
        gpu: &MetalGPU,
        a: *mut std::ffi::c_void,
        b: *mut std::ffi::c_void,
        c: *mut std::ffi::c_void,
        size: u32,
    ) -> Result<(), String> {
        unsafe {
            let func = Self::get_function(gpu, "add_kernel")?;
            let cmd_buffer = gpu.create_command_buffer()?;
            metal_dispatch_add(cmd_buffer, func, a, b, c, size);
            gpu.submit_commands(cmd_buffer)?;
            Ok(())
        }
    }

    /// Execute gelu kernel
    pub fn gelu(
        gpu: &MetalGPU,
        input: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        size: u32,
    ) -> Result<(), String> {
        unsafe {
            let func = Self::get_function(gpu, "gelu_kernel")?;
            let cmd_buffer = gpu.create_command_buffer()?;
            metal_dispatch_gelu(cmd_buffer, func, input, output, size);
            gpu.submit_commands(cmd_buffer)?;
            Ok(())
        }
    }

    /// Execute fused matmul+add+gelu kernel
    pub fn fused_matmul_add_gelu(
        gpu: &MetalGPU,
        a: *mut std::ffi::c_void,
        b: *mut std::ffi::c_void,
        bias: *mut std::ffi::c_void,
        c: *mut std::ffi::c_void,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), String> {
        unsafe {
            let func = Self::get_function(gpu, "fused_matmul_add_gelu_kernel")?;
            let cmd_buffer = gpu.create_command_buffer()?;
            metal_dispatch_fused_matmul_add_gelu(cmd_buffer, func, a, b, bias, c, m, n, k);
            gpu.submit_commands(cmd_buffer)?;
            Ok(())
        }
    }
}

// Metal FFI for kernel dispatch are provided by metal_stubs module
// In production, these would be replaced with actual Metal Objective-C bindings

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_loading() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = MetalGPU::new().unwrap();
        let func = MetalKernels::get_function(&gpu, "matmul_kernel");
        assert!(func.is_ok(), "Failed to load matmul kernel");
    }
}
