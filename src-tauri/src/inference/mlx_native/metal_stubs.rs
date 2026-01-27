// Metal stub implementations for testing
// In production, these would be replaced with actual Metal Objective-C bindings

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

static DEVICE_ID_COUNTER: AtomicUsize = AtomicUsize::new(100);
static BUFFER_ID_COUNTER: AtomicUsize = AtomicUsize::new(1000);

pub unsafe fn metal_is_available() -> bool {
    // Check if running on Apple Silicon
    cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")
}

pub unsafe fn metal_create_default_device() -> *mut c_void {
    let id = DEVICE_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    id as *mut c_void
}

pub unsafe fn metal_release_device(_device: *mut c_void) {
    // No-op for stub
}

pub unsafe fn metal_create_command_queue(_device: *mut c_void) -> *mut c_void {
    1001 as *mut c_void
}

pub unsafe fn metal_release_command_queue(_queue: *mut c_void) {
    // No-op for stub
}

pub unsafe fn metal_create_library(_device: *mut c_void) -> *mut c_void {
    2001 as *mut c_void
}

pub unsafe fn metal_release_library(_library: *mut c_void) {
    // No-op for stub
}

pub unsafe fn metal_create_buffer(_device: *mut c_void, _size: usize) -> *mut c_void {
    let id = BUFFER_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    // In a real implementation, this would allocate GPU memory
    // For now, just return a unique ID
    id as *mut c_void
}

pub unsafe fn metal_release_buffer(_buffer: *mut c_void) {
    // No-op for stub
}

pub unsafe fn metal_copy_to_gpu(_buffer: *mut c_void, _data: *const c_void, _size: usize) {
    // No-op for stub - in real implementation would copy to GPU
}

pub unsafe fn metal_copy_from_gpu(_buffer: *mut c_void, _data: *mut c_void, _size: usize) {
    // No-op for stub - in real implementation would copy from GPU
}

pub unsafe fn metal_create_command_buffer(_queue: *mut c_void) -> *mut c_void {
    3001 as *mut c_void
}

pub unsafe fn metal_commit_command_buffer(_buffer: *mut c_void) {
    // No-op for stub
}

pub unsafe fn metal_wait_command_buffer(_buffer: *mut c_void) {
    // No-op for stub
}

pub unsafe fn metal_create_function(
    _library: *mut c_void,
    _name: *const std::os::raw::c_char,
) -> *mut c_void {
    4001 as *mut c_void
}

pub unsafe fn metal_dispatch_matmul(
    _cmd_buffer: *mut c_void,
    _func: *mut c_void,
    _a: *mut c_void,
    _b: *mut c_void,
    _c: *mut c_void,
    _m: u32,
    _n: u32,
    _k: u32,
) {
    // No-op for stub
}

pub unsafe fn metal_dispatch_add(
    _cmd_buffer: *mut c_void,
    _func: *mut c_void,
    _a: *mut c_void,
    _b: *mut c_void,
    _c: *mut c_void,
    _size: u32,
) {
    // No-op for stub
}

pub unsafe fn metal_dispatch_gelu(
    _cmd_buffer: *mut c_void,
    _func: *mut c_void,
    _input: *mut c_void,
    _output: *mut c_void,
    _size: u32,
) {
    // No-op for stub
}

pub unsafe fn metal_dispatch_fused_matmul_add_gelu(
    _cmd_buffer: *mut c_void,
    _func: *mut c_void,
    _a: *mut c_void,
    _b: *mut c_void,
    _bias: *mut c_void,
    _c: *mut c_void,
    _m: u32,
    _n: u32,
    _k: u32,
) {
    // No-op for stub
}
