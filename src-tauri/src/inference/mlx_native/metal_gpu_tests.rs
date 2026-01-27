#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::metal_gpu::MetalGPU;

    #[test]
    fn test_metal_availability() {
        let available = MetalGPU::is_available();
        assert!(available, "Metal GPU should be available on Apple Silicon");
    }

    #[test]
    fn test_metal_device_creation() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = MetalGPU::new();
        assert!(gpu.is_ok(), "Failed to create Metal device");
    }

    #[test]
    fn test_metal_buffer_allocation() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = MetalGPU::new().unwrap();
        let buffer = gpu.create_buffer(1024 * 4);
        assert!(buffer.is_ok(), "Failed to allocate GPU buffer");
    }

    #[test]
    fn test_metal_command_buffer() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = MetalGPU::new().unwrap();
        let cmd_buffer = gpu.create_command_buffer();
        assert!(cmd_buffer.is_ok(), "Failed to create command buffer");
    }
}
