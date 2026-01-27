#[cfg(test)]
mod integration_tests {
    use crate::inference::mlx_native::compute_graph::{ComputeGraph, Operation};
    use crate::inference::mlx_native::gpu_buffer_pool::BufferPool;
    use crate::inference::mlx_native::gpu_graph_executor::GPUGraphExecutor;
    use crate::inference::mlx_native::metal_gpu::MetalGPU;
    use crate::inference::mlx_native::unified_memory::MLXArray;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_gpu_executor_with_compute_graph() {
        if !MetalGPU::is_available() {
            return;
        }

        let mut graph = ComputeGraph::new();
        let mm_id = graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![0, 1]);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(
                vec![1.0, 2.0, 3.0, 4.0],
                crate::inference::mlx_native::unified_memory::ArrayShape::Shape2D(2, 2),
            ),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(
                vec![5.0, 6.0, 7.0, 8.0],
                crate::inference::mlx_native::unified_memory::ArrayShape::Shape2D(2, 2),
            ),
        );

        let executor = GPUGraphExecutor::new().unwrap();
        let results = executor.execute(&graph, &inputs);

        assert!(results.is_ok(), "GPU executor should execute compute graph");
        assert!(results.unwrap().contains_key(&mm_id));
    }

    #[test]
    fn test_buffer_pool_with_multiple_allocations() {
        if !MetalGPU::is_available() {
            return;
        }

        let gpu = Arc::new(MetalGPU::new().unwrap());
        let pool = BufferPool::new(gpu, 10 * 1024);

        let _buf1 = pool.allocate(1024).unwrap();
        let _buf2 = pool.allocate(2048).unwrap();

        let stats = pool.statistics();
        assert!(stats.total_allocated >= 3072);
    }

    #[test]
    fn test_gpu_fallback_to_cpu_on_small_data() {
        if !MetalGPU::is_available() {
            return;
        }

        let mut graph = ComputeGraph::new();
        graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![0, 1]);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(
                vec![1.0, 2.0],
                crate::inference::mlx_native::unified_memory::ArrayShape::Shape1D(2),
            ),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(
                vec![3.0, 4.0],
                crate::inference::mlx_native::unified_memory::ArrayShape::Shape1D(2),
            ),
        );

        let executor = GPUGraphExecutor::new().unwrap();
        let results = executor.execute(&graph, &inputs);

        assert!(results.is_ok(), "GPU executor should handle small data");
    }

    #[test]
    fn test_metal_device_lifecycle() {
        if !MetalGPU::is_available() {
            return;
        }

        let gpu = MetalGPU::new().unwrap();
        let buffer = gpu.create_buffer(1024).unwrap();
        gpu.release_buffer(buffer);
        // Drop GPU - should cleanup resources
    }

    #[test]
    fn test_gpu_data_transfer() {
        if !MetalGPU::is_available() {
            return;
        }

        let gpu = MetalGPU::new().unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let buffer = gpu.create_buffer(data.len() * 4).unwrap();
        let copy_result = gpu.copy_to_gpu(buffer, &data);
        assert!(copy_result.is_ok());

        let mut result = vec![0.0; 4];
        let transfer_result = gpu.copy_from_gpu(buffer, &mut result);
        assert!(transfer_result.is_ok());

        gpu.release_buffer(buffer);
    }
}
