#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::gpu_buffer_pool::BufferPool;
    use crate::inference::mlx_native::metal_gpu::MetalGPU;
    use std::sync::Arc;

    #[test]
    fn test_buffer_pool_allocation() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = Arc::new(MetalGPU::new().unwrap());
        let pool = BufferPool::new(gpu, 1024 * 1024);

        let buf1 = pool.allocate(1024);
        assert!(buf1.is_ok(), "Failed to allocate buffer");
        assert_eq!(buf1.unwrap().size(), 1024);
    }

    #[test]
    fn test_buffer_pool_reuse() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = Arc::new(MetalGPU::new().unwrap());
        let pool = BufferPool::new(gpu, 1024 * 1024);

        let buf1 = pool.allocate(1024).unwrap();
        let ptr1 = buf1.ptr();
        pool.release(buf1);

        let buf2 = pool.allocate(1024).unwrap();
        let ptr2 = buf2.ptr();

        assert_eq!(ptr1, ptr2, "Buffer pool should reuse allocated buffers");
    }

    #[test]
    fn test_pool_statistics() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = Arc::new(MetalGPU::new().unwrap());
        let pool = BufferPool::new(gpu, 1024 * 1024);

        let _buf = pool.allocate(1024).unwrap();
        let stats = pool.statistics();
        assert!(stats.total_allocated > 0);
    }

    #[test]
    fn test_pool_clear() {
        if !MetalGPU::is_available() {
            return;
        }
        let gpu = Arc::new(MetalGPU::new().unwrap());
        let pool = BufferPool::new(gpu, 1024 * 1024);

        let _buf = pool.allocate(1024).unwrap();
        pool.clear();

        let stats = pool.statistics();
        assert_eq!(stats.total_allocated, 0);
    }
}
